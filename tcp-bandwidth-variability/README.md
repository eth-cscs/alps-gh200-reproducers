```bash
salloc --reservation=interactive_jobs -w nid[007384,007386] -t 30:00 -N2 -A csstaff --ntasks-per-node=1
salloc -p debug -w nid[005016-005017] -t 30:00 -N2 -A csstaff --ntasks-per-node=1


clariden:
sbatch -p debug -w nid[005016-005017] -A a-csstaff tcp-bw-var.sh
sbatch -p debug -w nid[005016-005018] -A a-csstaff tcp-bw-var.sh

eiger:
sbatch -p debug -w nid[002246-002247] -A csstaff -C mc -t 10:00 tcp-bw-var.sh


python3 plot-tcp-bw.py -o tcp-bw-var-2025-03-14.png --summary=results/2025-03-14/summary.txt results/2025-03-14/nj-*out
python3 plot-tcp-bw.py -o tcp-bw-var-2025-03-15.png --summary=results/2025-03-15/summary.txt results/2025-03-15/nj-*out

```



cat /sys/class/net/hsn0/queues/rx-0/rps_cpus
Receive Packet Steering

Poor Interrupt Affinity (IRQ Handling on a Single Core)
	•	If all network interrupts are handled by a single core, that core may become a bottleneck.
	•	This can cause the kernel to reschedule tasks frequently.

echo "f" | sudo tee /proc/irq/<IRQ_NUMBER>/smp_affinity
