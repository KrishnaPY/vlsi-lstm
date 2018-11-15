`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   13:22:58 04/13/2016
// Design Name:   dafir
// Module Name:   C:/XilinxWork/VLSIDSP/compare/dafir/test_dafir.v
// Project Name:  dafir
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: dafir
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module test_dafir;

	// Inputs
	reg [7:0] xnp1;
	reg clk;

	// Outputs
	wire [16:0] opy;

	// Instantiate the Unit Under Test (UUT)
	dafir uut (
		.opy(opy), 
		.xnp1(xnp1), 
		.clk_bit(clk)
	);
	initial begin
		clk = 0;
		forever #10 clk=~clk;
	end



	initial begin
		// Initialize Inputs
		xnp1 = 0;
		
		// Add stimulus here
		#120 xnp1 = -6;
		#160 xnp1 = 9;
		#160 xnp1 = 0;
		#160 xnp1 = 2;
		#160 xnp1 = 11;
		#160 xnp1 = 36;
		#160 xnp1 = 21;
		#160 xnp1 = 2;
		#160 xnp1 = 1;
		#160 xnp1 = 15;
		#160 xnp1 = 18;
        
		

	end
      
endmodule

