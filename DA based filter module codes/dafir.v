`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    10:17:14 04/13/2016 
// Design Name: 
// Module Name:    dafir 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module dafir(output [16:0] opy,
             input [7:0] xnp1,
             input clk_bit);
                                  

wire [9:0] xnm2, xnm3;

wire clk_byte;
wire [2:0] count;
wire [3:0] A;


// All the modules are instantiated
clk_gen clkgen1(.count(count), .byte_clk(clk_byte), .bit_clk(clk_bit));

da_prod_blk blk1(.sum(opy), .xn(), .xnm1(), .xnm2(xnm2), .xnm3(xnm3), .xnp1(xnp1), .wkl(A), .count(count), .clk_bit(clk_bit), .clk_byte(clk_byte));

weights wtblk1(.y(A), .count(count));

endmodule



// Inner Product Module start
module da_prod_blk(output reg [16:0] sum,
                   output reg [9:0] xn,
                   output reg [9:0] xnm1,
                   output reg [9:0] xnm2,
                   output reg [9:0] xnm3,                   
                   input [7:0] xnp1,
                   input [3:0] wkl,
                   input [2:0] count,
                   input clk_bit,
                   input clk_byte);


                   
initial begin
	sum=0;
	intsum=0;
	extra=0;

end

reg [6:0] extra;	

reg [9:0] s3,s5,s6,s9,s10,s12;
reg [9:0] s7,s11,s13,s14,s15;
reg [9:0] muxout;
reg [9:0] intsum;

initial begin
  xn=0;
  xnm1=0;
  xnm2=0;
  xnm3=0;
  s3=0;
  s5=0;
  s6=0;
  s7=0;
  s9=0;
  s10=0;
  s11=0;
  s12=0;
  s13=0;
  s14=0;
  s15=0;
end


always @ (posedge clk_bit) begin
	extra = extra>>1;
	extra[6] = intsum[0];
	if(count==0)	intsum=(~muxout)+{intsum[9],intsum[9:1]}+1;
	else intsum=muxout + {intsum[9],intsum[9:1]};
end	


always @ (posedge clk_byte) begin
	sum <= {intsum,extra};
	intsum<=0;
end
always @ (posedge clk_byte) begin

  xn<={{2{xnp1[7]}},xnp1};        //sum1
  xnm1<=xn;    					    //sum2
  xnm2<=xnm1;     					 //sum4
  xnm3<=xnm2;    						 //sum8
  s3<={{2{xnp1[7]}},xnp1}+xn;
  s5<={{2{xnp1[7]}},xnp1}+xnm1;
  s6<=s3;
  s7<={{2{xnp1[7]}},xnp1}+s3;
  s9<={{2{xnp1[7]}},xnp1}+xnm2;
  s10<=s5;
  s11<={{2{xnp1[7]}},xnp1}+s5;
  s12<=s6;
  s13<={{2{xnp1[7]}},xnp1}+s6;
  s14<=s7;
  s15<={{2{xnp1[7]}},xnp1}+s7;
  
 end


 
 always @(*) begin
   case (wkl)
     0: muxout <= 0;
     1: muxout <= xn;
     2: muxout <= xnm1;
     3: muxout <= s3;
     4: muxout <= xnm2;
     5: muxout <= s5;
     6: muxout <= s6;
     7: muxout <= s7;
     8: muxout <= xnm3;
     9: muxout <= s9;
    10: muxout <= s10;
    11: muxout <= s11;
    12: muxout <= s12;
    13: muxout <= s13;
    14: muxout <= s14;
    15: muxout <= s15;
  endcase
  
end 
  
endmodule



// Clock Generator Module
module clk_gen(output reg [2:0] count,
               output byte_clk,
               input bit_clk);
               

initial count=3'b000;

assign byte_clk = count[2];

always @(posedge bit_clk) count=count-1;


endmodule



// Weights  Block

module weights(output reg [3:0] y,
					input [2:0] count);
						

  
  parameter y4=8'b11110001,
				y3=8'b11110110,
				y2=8'b00001001,
				y1=8'b00000101;
  
initial begin
   y=0;
end
  
  
   always @ (*)  begin
  
    case (count)
    7: begin
       y[0]<=y4[0];
       y[1]<=y3[0];
       y[2]<=y2[0];
       y[3]<=y1[0];
       end
    
    6: begin
       y[0]<=y4[1];
       y[1]<=y3[1];
       y[2]<=y2[1];
       y[3]<=y1[1];
       end
       
    5: begin
       y[0]<=y4[2];
       y[1]<=y3[2];
       y[2]<=y2[2];
       y[3]<=y1[2];
       end
      
    4: begin
       y[0]<=y4[3];
       y[1]<=y3[3];
       y[2]<=y2[3];
       y[3]<=y1[3];
       end
       
    3: begin
       y[0]<=y4[4];
       y[1]<=y3[4];
       y[2]<=y2[4];
       y[3]<=y1[4];
       end

    2: begin
       y[0]<=y4[5];
       y[1]<=y3[5];
       y[2]<=y2[5];
       y[3]<=y1[5];
       end
  
    1: begin
       y[0]<=y4[6];
       y[1]<=y3[6];
       y[2]<=y2[6];
       y[3]<=y1[6];
       end
 
    0: begin
       y[0]<=y4[7];
       y[1]<=y3[7];
       y[2]<=y2[7];
       y[3]<=y1[7];
       end

  endcase
  
end


endmodule
