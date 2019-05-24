from pssh.clients import ParallelSSHClient
from os.path import join
import sys

rx_hosts = ['10.18.194.134', '10.18.194.99']
tx_hosts = ['10.18.194.165']
base_path = "/home/nvidia/daily_samples"
path = str(sys.argv[1]).replace("\\", "/")

rx = ParallelSSHClient(rx_hosts, user='root', password='nvidia')
tx = ParallelSSHClient(tx_hosts, user='root', password='nvidia')

output_rx = rx.run_command("mkdir -p {}".format(join(base_path, *path.split('/')[:-1]).replace("\\", "/")))
rx.join(output_rx)
output_rx = rx.run_command("recv_csi {}.dat".format(join(base_path, path).replace("\\", "/")))
output_tx = tx.run_command('send_Data wlan0 00:0e:8e:59:8a:aa 5000 1000')
tx.join(output_tx)
output = rx.run_command('sudo kill -9 $(pgrep -f recv_csi)')
