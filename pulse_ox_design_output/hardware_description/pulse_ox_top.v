// ===========================================
// Pulse Oximeter - Verilog Version
// File: pulse_ox_top.v
// ===========================================

module pulse_ox_top (
    input wire clk,
    input wire rst_n,
    
    // Sensor interface
    input wire [15:0] sensor_ir,
    input wire [15:0] sensor_red,
    input wire sensor_valid,
    
    // Output interface
    output reg [7:0] spo2_value,
    output reg [15:0] hr_value,
    output reg data_valid,
    
    // Debug
    output reg [15:0] debug_data
);

    // State definitions
    localparam IDLE = 3'b000,
               ACQUIRE = 3'b001,
               PROCESS = 3'b010,
               OUTPUT = 3'b011;
    
    reg [2:0] state;
    reg [6:0] sample_count;  // 0-127
    
    // Buffer memories
    reg [15:0] ir_buffer [0:99];
    reg [15:0] red_buffer [0:99];
    
    // Buffer pointers
    reg [6:0] write_ptr;
    reg [6:0] read_ptr;
    
    // Processing registers
    reg [31:0] ir_sum;
    reg [31:0] red_sum;
    reg [31:0] ir_ac_sum;
    reg [31:0] red_ac_sum;
    reg [15:0] ir_mean;
    reg [15:0] red_mean;
    
    integer i;
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            write_ptr <= 0;
            sample_count <= 0;
            data_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    data_valid <= 0;
                    if (sensor_valid) begin
                        state <= ACQUIRE;
                        write_ptr <= 0;
                        sample_count <= 0;
                    end
                end
                
                ACQUIRE: begin
                    if (sensor_valid) begin
                        ir_buffer[write_ptr] <= sensor_ir;
                        red_buffer[write_ptr] <= sensor_red;
                        
                        if (write_ptr == 99) begin
                            state <= PROCESS;
                            write_ptr <= 0;
                        end else begin
                            write_ptr <= write_ptr + 1;
                        end
                    end
                end
                
                PROCESS: begin
                    // Calculate sums
                    ir_sum = 0;
                    red_sum = 0;
                    
                    for (i = 0; i < 100; i = i + 1) begin
                        ir_sum = ir_sum + ir_buffer[i];
                        red_sum = red_sum + red_buffer[i];
                    end
                    
                    ir_mean = ir_sum / 100;
                    red_mean = red_sum / 100;
                    
                    // Calculate AC components
                    ir_ac_sum = 0;
                    red_ac_sum = 0;
                    
                    for (i = 0; i < 100; i = i + 1) begin
                        ir_ac_sum = ir_ac_sum + 
                            ((ir_buffer[i] > ir_mean) ? 
                             (ir_buffer[i] - ir_mean) : (ir_mean - ir_buffer[i]));
                        red_ac_sum = red_ac_sum + 
                            ((red_buffer[i] > red_mean) ? 
                             (red_buffer[i] - red_mean) : (red_mean - red_buffer[i]));
                    end
                    
                    state <= OUTPUT;
                end
                
                OUTPUT: begin
                    // Simplified SpO2 calculation
                    // spo2 = 110 - 25 * (red_ac/red_dc)/(ir_ac/ir_dc)
                    // Using approximations here
                    spo2_value <= 98;  // Placeholder
                    hr_value <= 72;    // Placeholder
                    data_valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
