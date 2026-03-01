-- ===========================================
-- Pulse Oximeter Digital Front-End
-- File: pulse_ox_frontend.vhd
-- ===========================================

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity PulseOximeterFrontend is
    Port (
        clk             : in  STD_LOGIC;
        reset           : in  STD_LOGIC;
        -- Sensor interface
        sensor_ir_data  : in  STD_LOGIC_VECTOR(15 downto 0);
        sensor_red_data : in  STD_LOGIC_VECTOR(15 downto 0);
        sensor_valid    : in  STD_LOGIC;
        -- Control
        sample_rate     : in  STD_LOGIC_VECTOR(15 downto 0);
        -- Output
        spo2_out        : out STD_LOGIC_VECTOR(7 downto 0);
        hr_out          : out STD_LOGIC_VECTOR(15 downto 0);
        data_ready      : out STD_LOGIC
    );
end PulseOximeterFrontend;

architecture Behavioral of PulseOximeterFrontend is
    
    -- Component declarations
    component MovingAverageFilter
        Generic (
            WIDTH : integer := 16;
            DEPTH : integer := 100
        );
        Port (
            clk     : in  STD_LOGIC;
            reset   : in  STD_LOGIC;
            data_in : in  STD_LOGIC_VECTOR(WIDTH-1 downto 0);
            valid   : in  STD_LOGIC;
            data_out: out STD_LOGIC_VECTOR(WIDTH-1 downto 0)
        );
    end component;
    
    component SpO2Calculator
        Port (
            clk         : in  STD_LOGIC;
            reset       : in  STD_LOGIC;
            ir_ac       : in  STD_LOGIC_VECTOR(15 downto 0);
            red_ac      : in  STD_LOGIC_VECTOR(15 downto 0);
            ir_dc       : in  STD_LOGIC_VECTOR(15 downto 0);
            red_dc      : in  STD_LOGIC_VECTOR(15 downto 0);
            spo2_result : out STD_LOGIC_VECTOR(7 downto 0)
        );
    end component;
    
    component HeartRateDetector
        Port (
            clk       : in  STD_LOGIC;
            reset     : in  STD_LOGIC;
            signal_in : in  STD_LOGIC_VECTOR(15 downto 0);
            valid_in  : in  STD_LOGIC;
            hr_out    : out STD_LOGIC_VECTOR(15 downto 0)
        );
    end component;
    
    -- Internal signals
    signal ir_filtered  : STD_LOGIC_VECTOR(15 downto 0);
    signal red_filtered : STD_LOGIC_VECTOR(15 downto 0);
    signal ir_dc        : STD_LOGIC_VECTOR(15 downto 0);
    signal red_dc       : STD_LOGIC_VECTOR(15 downto 0);
    signal ir_ac        : STD_LOGIC_VECTOR(15 downto 0);
    signal red_ac       : STD_LOGIC_VECTOR(15 downto 0);
    
    type state_type is (IDLE, ACQUIRE, PROCESSING, OUTPUT);
    signal state : state_type;
    
    signal sample_counter : unsigned(15 downto 0);
    constant SAMPLE_COUNT_MAX : unsigned(15 downto 0) := to_unsigned(100, 16);
    
begin
    
    -- Moving average filters for DC components
    IR_DC_FILTER: MovingAverageFilter
        generic map (WIDTH => 16, DEPTH => 100)
        port map (
            clk => clk,
            reset => reset,
            data_in => sensor_ir_data,
            valid => sensor_valid,
            data_out => ir_dc
        );
    
    RED_DC_FILTER: MovingAverageFilter
        generic map (WIDTH => 16, DEPTH => 100)
        port map (
            clk => clk,
            reset => reset,
            data_in => sensor_red_data,
            valid => sensor_valid,
            data_out => red_dc
        );
    
    -- AC component extraction (signal - DC)
    process(clk, reset)
    begin
        if reset = '1' then
            ir_ac <= (others => '0');
            red_ac <= (others => '0');
        elsif rising_edge(clk) then
            if sensor_valid = '1' then
                ir_ac <= std_logic_vector(signed(sensor_ir_data) - signed(ir_dc));
                red_ac <= std_logic_vector(signed(sensor_red_data) - signed(red_dc));
            end if;
        end if;
    end process;
    
    -- SpO2 calculation
    SPO2_CALC: SpO2Calculator
        port map (
            clk => clk,
            reset => reset,
            ir_ac => ir_ac,
            red_ac => red_ac,
            ir_dc => ir_dc,
            red_dc => red_dc,
            spo2_result => spo2_out
        );
    
    -- Heart rate detection
    HR_DETECT: HeartRateDetector
        port map (
            clk => clk,
            reset => reset,
            signal_in => ir_filtered,
            valid_in => sensor_valid,
            hr_out => hr_out
        );
    
    -- State machine for data acquisition
    process(clk, reset)
    begin
        if reset = '1' then
            state <= IDLE;
            sample_counter <= (others => '0');
            data_ready <= '0';
        elsif rising_edge(clk) then
            case state is
                when IDLE =>
                    data_ready <= '0';
                    if sensor_valid = '1' then
                        state <= ACQUIRE;
                        sample_counter <= (others => '0');
                    end if;
                
                when ACQUIRE =>
                    if sensor_valid = '1' then
                        if sample_counter = SAMPLE_COUNT_MAX - 1 then
                            state <= PROCESSING;
                        else
                            sample_counter <= sample_counter + 1;
                        end if;
                    end if;
                
                when PROCESSING =>
                    state <= OUTPUT;
                
                when OUTPUT =>
                    data_ready <= '1';
                    state <= IDLE;
            end case;
        end if;
    end process;
    
end Behavioral;
