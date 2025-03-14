#!/usr/bin/env python3
#
# Jonathan Coles <jonathan.coles@cscs.ch>

import argparse

import pylab as pl
import numpy as np
import pandas as pd
import os
import matplotlib.dates as mdates

from glob import glob
from datetime import datetime
from collections import defaultdict

def tcp_bw_var_dataframe(fname):

    print(f"Reading {fname}")

    scale = {
            'Gbits/sec': 1,
            'Mbits/sec': 1e-3,
            'Kbits/sec': 1e-6,
            'bits/sec': 1e-9
            }

    client = os.path.splitext(fname.split('-')[-1])[0]

    D = defaultdict(list)

    mode = 'wait for bw'

    server=None
    with open(fname, 'rt') as fp:
        for line in fp:
            line = line.strip()
            if not line: continue

            toks = line.split()

            if mode == 'wait for bw':

                if 'server:' in line:
                    server = dict(hostname = toks[1], xname = toks[2])

                if 'client:' in line:
                    client = dict(hostname = toks[1], xname = toks[2])

                if 'Interval' in line:
                    mode = 'read bw'

            if mode == 'read bw':

                if 'bits/sec' in line:
                    ts        = toks[0]
                    int0,int1 = map(float, toks[-9].split('-'))
                    bitrate   = float(toks[-5])
                    units     = toks[-4]

                    bitrate *= scale[units]

                    D['Bitrate (Gbits/sec)'].append(bitrate)
                    D['Interval (sec)'].append(int0)
                    D['Timestamp'].append(ts)

                if '- - - -' in line: 
                    mode = 'skip to next'

            if mode == 'skip to next':
                if 'iperf Done.' in line:
                    mode = 'wait for bw'

    df = pd.DataFrame(D)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    return {'client': client, 'server': server, 'iperf-df': df}
            
def plot_tcp_bw_var(fnames_tcp_bw_var, fname_image, args):

    nr = 1
    nc = 1
    fig, axes = pl.subplots(nrows=nr, ncols=nc, figsize=(nc*12, nr*8), squeeze=False) #, gridspec_kw={'width_ratios': [8, 1]})
    pl.subplots_adjust(top=0.65, hspace=0.2)

    #pl.suptitle(plot_title)

    author_name = 'Jonathan Coles'
    author_email = 'jonathan.coles@cscs.ch'

    author = f'{author_name} <{author_email}>'
    date_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    plot_summary='''
        iperf3 bandwidth test using clariden (x1100) and eiger (x1004). Each line represents 20 consecutive runs of 20 seconds each.

        Client and server were pinned on both cpu and network interface using --affinity=10 --bind-dev=hsn0

        On Clariden one test uses two nodes within the same blade, while the other uses the same node in two neighboring slots (same chassis).
        The latter case should mean the nodes are communicating using a single switch.
    '''

    t = f'''
        {plot_summary}
        Generated on: {date_time}
        Author: {author}
        '''

    header_text = pl.text(0.05, 0.95, t, ha='left', va='top', transform=fig.transFigure, fontsize=9)
    header_text.set_in_layout(False)

    pl.suptitle('Network Bandwidth')

    ax = axes[0,0]

    colors = {}

    xmin,xmax=np.inf, -np.inf

    for fname in sorted(fnames_tcp_bw_var):
        D = tcp_bw_var_dataframe(fname)
        client = D['client']['xname']
        server = D['server']['xname']
        label = f'{server} (server) \n{client} (client)'
        df = D['iperf-df']

        #X0 = df['Interval (sec)']
        X0 = (df['Timestamp'] - df['Timestamp'][0]).dt.total_seconds()
        Y0 = df['Bitrate (Gbits/sec)'] 
        l,= ax.plot(X0, Y0,  ls='-', mec=None, lw=0.5, alpha=1.0, label=label)

        #moving_mean, plus_1_sigma, minus_1_sigma = calculate_moving_mean_and_sigma(Y0, 60)
        #l,= ax.plot(X0[59:], moving_mean,  ls='-', mec=None, lw=0.5, alpha=1.0, label=label)
        #ax.fill_between(X0[59:], minus_1_sigma, plus_1_sigma, color=l.get_color(), alpha=0.1)


        #l,= ax.plot(X0, Y0,  marker='s', markersize=2,ls='-', mec=None, lw=0.3, alpha=0.5, label=client)
        #ax.plot(X0, Y0, c=l.get_color(), marker='s', mec='w', markersize=5,mfc=l.get_color(), ls='', lw=0.1)
        xmin = min(X0.min(), xmin)
        xmax = max(X0.max(), xmax)

        colors.setdefault(client, l.get_color())

    #ax.set_xticks(range(int(xmin), int(xmax + 1), 20))

# Enable the grid on the x-axis
    #ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    ax.legend(loc='lower left', bbox_to_anchor=(0.00, 1), fontsize=9, ncol=3)
    #pl.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)

    #ax.set_xticks(ax.get_xticks()[::20])

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')  # Optional: aligns better when rotated
    ax.set_xlabel('Time [seconds]')
    ax.set_ylabel('Bandwidth [Gbits/s]')
    #ax.set_yscale('symlog', base=2)

#    for i,ax in np.ndenumerate(axes):
#        ax.set_xlim(xmin.total_seconds(),xmax.total_seconds())

    #ax = axes[0,1]

    #for c,y in zip(['g', 'b', 'r'], [Y0,Y1,Y2]):
    #    counts, bin_edges = np.histogram(y, bins=200)
    #    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #    ax.plot(counts / np.sum(counts), bin_edges[:-1], c=c, ls='-', lw=0.5)

    #ax.set_xlabel('Fraction of Occurances')
    #ax.set_yscale('symlog', base=2)
    ##ax.set_xlim(0,1)
    #ax.yaxis.set_tick_params(labelleft=False)

    #pl.subplots_adjust(wspace=0, hspace=0)

    #ax = axes[0,0]
    #legend = [
    #    ['iopsstor scratch',  pl.Line2D([0], [0], lw=5, color='g')],
    #    ['capstor scratch',   pl.Line2D([0], [0], lw=5, color='r')],
    #    ['vast home',         pl.Line2D([0], [0], lw=5, color='b')],
    #]

    #l, h = list(zip(*legend))
    #ax.legend(h, l, frameon=False, loc='upper right', ncol=4, fontsize=8, bbox_to_anchor=(1.0, 1.0))
    #ax.legend()


    pl.savefig(fname_image)

def calculate_moving_mean_and_sigma(data, window_size):
    """
    Given a list or array of data, return three lists:
    - Moving mean of the data
    - +1 sigma deviation (mean + standard deviation)
    - -1 sigma deviation (mean - standard deviation)

    Parameters:
    data (list or np.array): Input data
    window_size (int): Size of the moving window

    Returns:
    tuple: (moving_mean, plus_1_sigma, minus_1_sigma)
    """
    # Convert data to a numpy array if it isn't already
    data = np.asarray(data)

    # Use a rolling window view for efficient computation
    shape = (len(data) - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # Compute moving mean and standard deviation
    moving_mean = np.mean(windows, axis=1)
    std_dev = np.std(windows, axis=1)

    # Calculate +1 sigma and -1 sigma deviations
    plus_1_sigma = moving_mean + std_dev
    minus_1_sigma = moving_mean - std_dev

    return moving_mean, plus_1_sigma, minus_1_sigma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot system jitter using the output summary of the sysjitter utility')
    # parser.add_argument('--summary_file', type=str, required=True, help='Path to sysjitter summary file')
    # parser.add_argument('--freezable_file', type=str, help='Path to freezable.out file')
    #parser.add_argument('--dir',         default='', dest='input_dir', type=str, required=False, help='Path to output image')
    #parser.add_argument('--iperf-bw',    default='*client-nid??????.out', dest='fname_iperf_bw', type=str, required=False, help='Path to output image')
    #parser.add_argument('--client-top',  default='*client-nid??????-top.out', dest='fname_client_top', type=str, required=False, help='Path to output image')
    #parser.add_argument('--client-free', default='*client-nid??????-free.out', dest='fname_client_free', type=str, required=False, help='Path to output image')
    parser.add_argument('-o', dest='fname_image', type=str, required=True, help='Path to output image')
    # parser.add_argument('--author_name', type=str, required=True, help='Name of the author')
    # parser.add_argument('--author_email', type=str, required=True, help='Email of the author')
    # parser.add_argument('--plot_title', type=str, default='System jitter', help='Title of the plot')
    # parser.add_argument('--plot_summary', type=str, default='System jitter as recorded on a node using ./sysjitter --runtime 300 --verbose 300', help='Summary string of the plot')

    args, remaining_args = parser.parse_known_args()

    fnames_client_top = []
    fnames_client_free = []
    fnames_client_counters = []
    fnames_iperf_bw = []

    for filename in remaining_args:
        if filename.endswith('-free.out'):
            fnames_client_free.append(filename)
        elif filename.endswith('-top.out'):
            fnames_client_top.append(filename)
        elif filename.endswith('-cntrs.out'):
            fnames_client_counters.append(filename)
        else:
            fnames_iperf_bw.append(filename)

    #print( args.fname_iperf_bw, args.fname_client_top, args.fname_client_free)

    #fnames_iperf_bw    = glob(os.path.join(args.input_dir, args.fname_iperf_bw))
    #fnames_client_top  = glob(os.path.join(args.input_dir, args.fname_client_top))
    #fnames_client_free = glob(os.path.join(args.input_dir, args.fname_client_free))


    plot_tcp_bw_var(
        fnames_iperf_bw,
        args.fname_image,
        args
    )
