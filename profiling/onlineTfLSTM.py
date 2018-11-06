class LSTM_TF(object):
    def __init__(self, inp_dim, out_dim, state_dim, bptt_truncate=4):
        self.bptt_truncate = bptt_truncate
        
        # Construct the computation graph
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            """
            Setup placeholders for inputs and initialize the weights
            """
            # Define placeholders for input, output
            x_words = tf.placeholder(tf.int32, [None])
            y_words = tf.placeholder(tf.int32, [None])
            
            # Define the weights of the graph
            self.c = tf.Variable(tf.zeros(shape=(state_dim,1)))
            self.s = tf.Variable(tf.zeros(shape=(state_dim,1)))
            
            self.U = tf.Variable(tf.random_uniform(shape=(4, state_dim, inp_dim),
                                                   minval=-np.sqrt(1./inp_dim), 
                                                   maxval=np.sqrt(1./inp_dim)))
            self.W = tf.Variable(tf.random_uniform(shape=(4, state_dim, state_dim),
                                                   minval=-np.sqrt(1./state_dim), 
                                                   maxval=np.sqrt(1./state_dim)))
            self.b = tf.Variable(tf.ones(shape=(4, state_dim, 1)))
                        
            self.V = tf.Variable(tf.random_uniform(shape=(out_dim, state_dim),
                                                   minval=-np.sqrt(1./state_dim), 
                                                   maxval=np.sqrt(1./state_dim)))
            self.d = tf.Variable(tf.ones(shape=(out_dim, 1)))
            
            # Define the input parameter for RMSPROP
            learn_r = tf.placeholder(tf.float32)
            decay_r = tf.placeholder(tf.float32)
            
            # Define the variable to hold the adaptive learning rates
            self.mU = tf.Variable(tf.zeros(shape=self.U.shape))
            self.mW = tf.Variable(tf.zeros(shape=self.W.shape))
            self.mb = tf.Variable(tf.zeros(shape=self.b.shape))
            
            self.mV = tf.Variable(tf.zeros(shape=self.V.shape))
            self.md = tf.Variable(tf.zeros(shape=self.d.shape))
            
            global_init = tf.global_variables_initializer()
            
            """
            Dynamic forward pass using tf.scan
            """
            # Define the forward step for each word
            def forward_step(acc, word):
                c, s, output = acc
                
                # LSTM layer
                i = tf.sigmoid(tf.reshape(self.U[0,:,word], (-1,1)) + tf.matmul(self.W[0], s) + self.b[0])
                f = tf.sigmoid(tf.reshape(self.U[1,:,word], (-1,1)) + tf.matmul(self.W[1], s) + self.b[1])
                o = tf.sigmoid(tf.reshape(self.U[2,:,word], (-1,1)) + tf.matmul(self.W[2], s) + self.b[2])
                g =    tf.tanh(tf.reshape(self.U[3,:,word], (-1,1)) + tf.matmul(self.W[3], s) + self.b[3])
                
                c = f*c + g*i
                s = tf.tanh(c)*o
                
                # Output calculation
                output = tf.matmul(self.V, s) + self.d
                
                return [c, s, output]
            
            # Step through the sequence of input words, each one at a time
            ce_init = [self.c, self.s, tf.zeros(shape=(out_dim,1))]
            results = tf.scan(forward_step, x_words, ce_init)

            outputs = results[2]
            update_c = self.c.assign(results[0][-1])
            update_s = self.s.assign(results[1][-1])
            
            """
            Compute derivatives and nudge the weights
            """
            # Compute the error using cross entropy
            errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs[..., 0], labels=y_words)
            errors = tf.reduce_mean(errors)
            
            dU = tf.gradients(errors, self.U)[0]
            dW = tf.gradients(errors, self.W)[0]
            db = tf.gradients(errors, self.b)[0]  
            
            dV = tf.gradients(errors, self.V)[0]
            dd = tf.gradients(errors, self.d)[0]
            
            # Update rmsprop learning rates
            update_mu = self.mU.assign(decay_r * self.mU + (1 - decay_r) * dU ** 2)
            update_mw = self.mW.assign(decay_r * self.mW + (1 - decay_r) * dW ** 2)
            update_mb = self.mb.assign(decay_r * self.mb + (1 - decay_r) * db ** 2)
            
            update_mv = self.mV.assign(decay_r * self.mV + (1 - decay_r) * dV ** 2)
            update_md = self.md.assign(decay_r * self.md + (1 - decay_r) * dd ** 2)           

            # Nudge the weights using the updated learning rates
            nudge_u = self.U.assign(self.U - learn_r*dU/tf.sqrt(self.mU + 1e-6))
            nudge_w = self.W.assign(self.W - learn_r*dW/tf.sqrt(self.mW + 1e-6))
            nudge_b = self.b.assign(self.b - learn_r*db/tf.sqrt(self.mb + 1e-6))
            
            nudge_v = self.V.assign(self.V - learn_r*dV/tf.sqrt(self.mV + 1e-6))   
            nudge_d = self.d.assign(self.d - learn_r*dd/tf.sqrt(self.md + 1e-6))
            
            reset_c = self.c.assign(tf.zeros(shape=(state_dim,1)))
            reset_s = self.s.assign(tf.zeros(shape=(state_dim,1)))            
                
            # The function to nudge the weight based on the pair of sequences x and y
            def backpropagate_through_time(x, y, learning_rate):
                results = self.session.run([reset_c, reset_s, # re-initialize cell and state to zeros
                                            errors,           # run the operation to compute the loss
                                            update_mu, update_mw, update_mb, update_mv, update_md, # update rmsprop learning rates
                                            nudge_v,   nudge_u,   nudge_w,   nudge_b,   nudge_d,   # compute derivatives and nudge the weights
                                            update_c, update_s],                                   # update the current state of the cell
                                           feed_dict={x_words: x, y_words: y, learn_r: learning_rate, decay_r: 0.9})
                return results[2]
            self.backpropagate_through_time = backpropagate_through_time
            
            """
            Other functions
            """
            # The prediction function, which only compute outputs without differentiation stuff
            def predict(x):
                pred_outputs = self.session.run([outputs,             # run the operation to compute the outputs
                                                 update_c, update_s], # update the current state of the cell
                                                feed_dict={x_words: x})[0]
                pred_outputs = tf.nn.softmax(pred_outputs[..., 0])
                return pred_outputs
            self.predict = predict
            
            # The function to manually reset the state to zeros, useful operation at the start of each sequence generation
            def reset_state():
                results = self.session.run([reset_c, reset_s])
            self.reset_state = reset_state
            
        self.session = tf.Session(graph=self.graph)
        self.session.run(global_init)
        
    def fit(self, X_train, y_train, epoch = 3, learning_rate = 0.01):
        indices = range(len(X_train))
        
        with self.session.as_default():
            for _ in xrange(epoch):
                np.random.shuffle(indices)
                smooth_loss = 0
                
                print "Epoch #" + str(_) + " started"
                
                for i in xrange(len(X_train)):
                    x = X_train[indices[i]]
                    y = y_train[indices[i]]
                    
                    errors = self.backpropagate_through_time(x, y, learning_rate)
                    smooth_loss = (errors + smooth_loss*i)/(i+1)
                    
                    if i%20000 == 0:
                        print "Example " + str(i) + ", Loss " + str(smooth_loss)
                
                print "Epoch #" + str(_) + " completed, Loss " + str(smooth_loss) + '\n'

