--BSS and MSS firmware download
info = debug.getinfo(1,'S');
file_path = (info.source);
file_path = string.gsub(file_path, "@","");
file_path = string.gsub(file_path, "DataCaptureDemo_xWR.lua","");
fw_path   = file_path.."..\\..\\rf_eval_firmware"


--ADC_Data file and Raw file and PacketReorder utitlity log file path
data_path     = "C:\\Users\\mmwave\\Documents\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\PostProc"
adc_data_path = "C:\\Users\\mmwave\\Documents\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin"
Raw_data_path = "C:\\Users\\mmwave\\Documents\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
pkt_log_path  = "C:\\Users\\mmwave\\Documents\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\pktlogfile.txt"

-- Directory to collect the data samples
samples_directory = "C:\\Users\\mmwave\\Documents\\mmwave_datacollection\\adc_data.bin"

-- ar1.ProfileConfig(0, 77, 100, 6, 60, 0, 0, 0, 0, 0, 0, 60.012, 0, 256, 10000, 0, 0, 30)

-- ar1.ProfileConfig(0, 77, 60, 6, 60, 0, 0, 0, 0, 0, 0, 15.015, 0, 256, 5000, 0, 0, 30)

-- -- if (ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 1, 0) == 0) then
-- if (ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 1, 0) == 0) then
--     WriteToLog("ChirpConfig Success\n", "green")
-- else
--     WriteToLog("ChirpConfig failure\n", "red")
-- end
-- RSTD.Sleep(1000)

-- if (ar1.FrameConfig(0, 0, 200, 230, 33, 0, 1) == 0) then
--     WriteToLog("FrameConfig Success\n", "green")
-- else
--     WriteToLog("FrameConfig failure\n", "red")
-- end
-- RSTD.Sleep(1000)

-- -- select Device type
-- if (ar1.SelectCaptureDevice("DCA1000") == 0) then
--     WriteToLog("SelectCaptureDevice Success\n", "green")
-- else
--     WriteToLog("SelectCaptureDevice failure\n", "red")
-- end
-- RSTD.Sleep(1000)

-- --DATA CAPTURE CARD API
-- if (ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098) == 0) then
--     WriteToLog("CaptureCardConfig_EthInit Success\n", "green")
-- else
--     WriteToLog("CaptureCardConfig_EthInit failure\n", "red")
-- end
-- RSTD.Sleep(1000)

-- --AWR12xx or xWR14xx-1, xWR16xx or xWR18xx or xWR68xx- 2 (second parameter indicates the device type)
-- if ((partId == 1642) or (partId == 1843) or (partId == 6843)) then
--     if (ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30) == 0) then
--         WriteToLog("CaptureCardConfig_Mode Success\n", "green")
--     else
--         WriteToLog("CaptureCardConfig_Mode failure\n", "red")
--     end
-- elseif ((partId == 1243) or (partId == 1443)) then
--     if (ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30) == 0) then
--         WriteToLog("CaptureCardConfig_Mode Success\n", "green")
--     else
--         WriteToLog("CaptureCardConfig_Mode failure\n", "red")
--     end
-- end
-- RSTD.Sleep(1000)

-- if (ar1.CaptureCardConfig_PacketDelay(25) == 0) then
--     WriteToLog("CaptureCardConfig_PacketDelay Success\n", "green")
-- else
--     WriteToLog("CaptureCardConfig_PacketDelay failure\n", "red")
-- end
-- RSTD.Sleep(1000)

user = "kick"
os.execute("mkdir C:\\Users\\mmwave\\Desktop\\daily_samples\\" .. os.date('%m_%d\\') .. user)
for i= 5,1,-1
do
--Start Record ADC data
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(1000)

--Trigger frame
    file_path = os.date('%m_%d\\') .. user .. os.date('\\%H_%M_%S')
    csi_cmd = "pythonw.exe C:\\Users\\mmwave\\Desktop\\collect_csi.py " .. file_path
    io.popen(csi_cmd)
    RSTD.Sleep(1000)
    ar1.StartFrame()
    
    WriteToLog("---- Go ----\n", "red")
    os.execute("C:\\Users\\mmwave\\source\\repos\\ConsoleApplication1\\Debug\\ConsoleApplication1.exe")

    RSTD.Sleep(10000)
    print('Sample No: %A', i)

--Packet reorder utility processing the Raw_ADC_data
    WriteToLog("Please wait for a few seconds for Packet reorder utility processing .....!!!! \n", "green")
--local x = os.clock()
    ar1.PacketReorderZeroFill(Raw_data_path, adc_data_path, pkt_log_path)
--print(string.format("elapsed time: %.2f\n", os.clock() - x))\
-- RSTD.Sleep(10000)
-- WriteToLog("Packet reorder utility processing done.....!!!! \n", "green")

-- Copy collected data to a user defined folder
    os.rename('C:\\Users\\mmwave\\Documents\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin', 'C:\\Users\\mmwave\\Desktop\\daily_samples\\' .. file_path .. '.bin')
-- WriteToLog("Sample saved to directory")
end
