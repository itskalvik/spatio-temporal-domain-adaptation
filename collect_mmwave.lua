--ADC_Data file and Raw file and PacketReorder utitlity log file path
adc_data_path = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin"
Raw_data_path = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
pkt_log_path  = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\pktlogfile.txt"
buzzer        = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\Scripts\\buzzer.exe"
dataset_path  = "C:\\Users\\mmwave\\Desktop\\mmwave_data\\"
lfs = require "lfs"

user = "prabhu"
os.execute("mkdir " .. dataset_path .. os.date('%m_%d\\') .. user)

RSTD.Sleep(5000)

for i = 15,1,-1
do
    WriteToLog("Samples Left:" .. i .. "\n", "red")

    --Start Record ADC data
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(1000)

    --Trigger frame
    file_path = os.date('%m_%d\\') .. user .. os.date('\\%H_%M_%S')
    ar1.StartFrame()
    os.execute(buzzer)
    RSTD.Sleep(10000)

    --Check file size
    size = lfs.attributes(Raw_data_path , 'size')
    if(size ~= 190227698) then
      WriteToLog("Raw_data size:" .. size .. "\n", "red")
      os.execute(buzzer)
      os.execute(buzzer)
      do return end
    end

    --Packet reorder utility processing the Raw_ADC_data
    ar1.PacketReorderZeroFill(Raw_data_path, dataset_path .. file_path .. ".bin" , pkt_log_path)
end

--Signal end of script
os.execute(buzzer)
os.execute(buzzer)
os.execute(buzzer)
