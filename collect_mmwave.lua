--ADC_Data file and Raw file and PacketReorder utitlity log file path
data_path     = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\PostProc"
adc_data_path = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin"
Raw_data_path = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
pkt_log_path  = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\pktlogfile.txt"
lfs = require "lfs"

user = "jon_doe"
os.execute("mkdir C:\\Users\\mmwave\\Desktop\\mmwave_data\\" .. os.date('%m_%d\\') .. user)

RSTD.Sleep(5000)

for i= 20,1,-1
do
--Start Record ADC data
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(1000)

--Trigger frame
    file_path = os.date('%m_%d\\') .. user .. os.date('\\%H_%M_%S')
    io.popen("pythonw.exe C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\Scripts\\collect_csi.py " .. file_path)
    RSTD.Sleep(1000)
    ar1.StartFrame()

    os.execute("C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\Scripts\\buzzer.exe")
    WriteToLog("---- Go ----\n", "red")

    RSTD.Sleep(10000)
    print('Sample No: %A', i)

--Packet reorder utility processing the Raw_ADC_data
    WriteToLog("Please wait for a few seconds for Packet reorder utility processing .....!!!! \n", "green")
    ar1.PacketReorderZeroFill(Raw_data_path, adc_data_path, pkt_log_path)

--Check file size
    size = lfs.attributes('C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin', 'size')
    if(size ~= 188416000) then
      os.execute("C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\Scripts\\buzzer.exe")
      os.execute("C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\Scripts\\buzzer.exe")
      break
    end

-- Copy collected data to a user defined folder
    os.rename('C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin', 'C:\\Users\\mmwave\\Desktop\\mmwave_data\\' .. file_path .. '.bin')
end
