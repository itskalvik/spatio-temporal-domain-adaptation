from pssh.clients import ParallelSSHClient

rx_hosts = ['10.18.194.134', '10.18.194.99']
tx_hosts = ['10.18.194.165']

rx = ParallelSSHClient(rx_hosts, user='root', password='nvidia')
tx = ParallelSSHClient(tx_hosts, user='root', password='nvidia')


output_tx = tx.run_command('start_hostapd.sh')
tx.join(output_tx)

output_rx = rx.run_command('start_recv.sh')
rx.join(output_rx)