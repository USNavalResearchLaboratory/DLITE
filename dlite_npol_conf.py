#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Dlite NPol
# Generated: Tue Feb  5 15:23:59 2019
##################################################


from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import fft
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from optparse import OptionParser
import time,sys,os

exec(open(sys.argv[1]).read())
tstart=int(time.time())
outdir=os.path.join(OUTDIR,str(tstart))
os.system('mkdir '+outdir)
os.system('cp '+sys.argv[1]+' '+os.path.join(outdir,'SETUP'))


class dlite_npol(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Dlite NPol")

        ##################################################
        # Variables
        ##################################################
        self.tint = tint = TINT
        self.samp_rate = samp_rate = BANDWIDTH
        self.lfft = lfft = NCHAN
        self.npol = npol = NPOL
        self.nav = nav = int(tint*samp_rate/lfft)
        self.gain = gain = GAIN
        self.freq0 = freq0 = FREQ
        self.rootname = rootname = os.path.join(outdir,'visfile')

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("addr0=192.168.10.2,addr1=192.168.20.2,addr2=192.168.30.2,addr3=192.168.40.2", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		otw_format='sc16',
        		channels=range(8),
        	),
        )
        self.uhd_usrp_source_0.set_clock_source('external', 0)
        self.uhd_usrp_source_0.set_time_source('external', 0)
        self.uhd_usrp_source_0.set_subdev_spec('A:A A:B', 0)
        self.uhd_usrp_source_0.set_clock_source('external', 1)
        self.uhd_usrp_source_0.set_time_source('external', 1)
        self.uhd_usrp_source_0.set_subdev_spec('A:A A:B', 1)
        self.uhd_usrp_source_0.set_clock_source('external', 2)
        self.uhd_usrp_source_0.set_time_source('external', 2)
        self.uhd_usrp_source_0.set_subdev_spec('A:A A:B', 2)
        self.uhd_usrp_source_0.set_clock_source('external', 3)
        self.uhd_usrp_source_0.set_time_source('external', 3)
        self.uhd_usrp_source_0.set_subdev_spec('A:A A:B', 3)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        self.uhd_usrp_source_0.set_center_freq(freq0, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_antenna('RX1', 0)
        self.uhd_usrp_source_0.set_center_freq(freq0, 1)
        self.uhd_usrp_source_0.set_gain(gain, 1)
        self.uhd_usrp_source_0.set_antenna('RX2', 1)
        self.uhd_usrp_source_0.set_center_freq(freq0, 2)
        self.uhd_usrp_source_0.set_gain(gain, 2)
        self.uhd_usrp_source_0.set_antenna('RX1', 2)
        self.uhd_usrp_source_0.set_center_freq(freq0, 3)
        self.uhd_usrp_source_0.set_gain(gain, 3)
        self.uhd_usrp_source_0.set_antenna('RX2', 3)
        self.uhd_usrp_source_0.set_center_freq(freq0, 4)
        self.uhd_usrp_source_0.set_gain(gain, 4)
        self.uhd_usrp_source_0.set_antenna('RX1', 4)
        self.uhd_usrp_source_0.set_center_freq(freq0, 5)
        self.uhd_usrp_source_0.set_gain(gain, 5)
        self.uhd_usrp_source_0.set_antenna('RX2', 5)
        self.uhd_usrp_source_0.set_center_freq(freq0, 6)
        self.uhd_usrp_source_0.set_gain(gain, 6)
        self.uhd_usrp_source_0.set_antenna('RX1', 6)
        self.uhd_usrp_source_0.set_center_freq(freq0, 7)
        self.uhd_usrp_source_0.set_gain(gain, 7)
        self.uhd_usrp_source_0.set_antenna('RX2', 7)
        self.blocks_stream_to_vector_1x = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_1y = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_2x = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_2y = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_3x = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_3y = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_4x = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.blocks_stream_to_vector_4y = blocks.stream_to_vector(gr.sizeof_gr_complex*1, lfft)
        self.fft_vxx_1x = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_1y = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_2x = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_2y = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_3x = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_3y = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_4x = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.fft_vxx_4y = fft.fft_vcc(lfft, True, (window.blackmanharris(lfft)), True, 1)
        self.blocks_multiply_conjugate_cc_12xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_13xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_14xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_23xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_24xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_34xx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_12yy = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_13yy = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_14yy = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_23yy = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_24yy = blocks.multiply_conjugate_cc(lfft)
        self.blocks_multiply_conjugate_cc_34yy = blocks.multiply_conjugate_cc(lfft)
        if(npol>2):
            self.blocks_multiply_conjugate_cc_12xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_13xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_14xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_23xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_24xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_34xy = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_12yx = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_13yx = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_14yx = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_23yx = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_24yx = blocks.multiply_conjugate_cc(lfft)
            self.blocks_multiply_conjugate_cc_34yx = blocks.multiply_conjugate_cc(lfft)
        self.blocks_integrate_xx_12xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_13xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_14xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_23xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_24xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_34xx = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_12yy = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_13yy = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_14yy = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_23yy = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_24yy = blocks.integrate_cc(nav, lfft)
        self.blocks_integrate_xx_34yy = blocks.integrate_cc(nav, lfft)
        if(npol>2):
            self.blocks_integrate_xx_12xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_13xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_14xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_23xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_24xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_34xy = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_12yx = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_13yx = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_14yx = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_23yx = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_24yx = blocks.integrate_cc(nav, lfft)
            self.blocks_integrate_xx_34yx = blocks.integrate_cc(nav, lfft)
        self.blocks_file_sink_12xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_12_xx.bin', False)
        self.blocks_file_sink_12xx.set_unbuffered(False)
        self.blocks_file_sink_13xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_13_xx.bin', False)
        self.blocks_file_sink_13xx.set_unbuffered(False)
        self.blocks_file_sink_14xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_14_xx.bin', False)
        self.blocks_file_sink_14xx.set_unbuffered(False)
        self.blocks_file_sink_23xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_23_xx.bin', False)
        self.blocks_file_sink_23xx.set_unbuffered(False)
        self.blocks_file_sink_24xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_24_xx.bin', False)
        self.blocks_file_sink_24xx.set_unbuffered(False)
        self.blocks_file_sink_34xx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_34_xx.bin', False)
        self.blocks_file_sink_34xx.set_unbuffered(False)
        self.blocks_file_sink_12yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_12_yy.bin', False)
        self.blocks_file_sink_12yy.set_unbuffered(False)
        self.blocks_file_sink_13yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_13_yy.bin', False)
        self.blocks_file_sink_13yy.set_unbuffered(False)
        self.blocks_file_sink_14yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_14_yy.bin', False)
        self.blocks_file_sink_14yy.set_unbuffered(False)
        self.blocks_file_sink_23yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_23_yy.bin', False)
        self.blocks_file_sink_23yy.set_unbuffered(False)
        self.blocks_file_sink_24yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_24_yy.bin', False)
        self.blocks_file_sink_24yy.set_unbuffered(False)
        self.blocks_file_sink_34yy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_34_yy.bin', False)
        self.blocks_file_sink_34yy.set_unbuffered(False)
        if(npol>2):
            self.blocks_file_sink_12xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_12_xy.bin', False)
            self.blocks_file_sink_12xy.set_unbuffered(False)
            self.blocks_file_sink_13xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_13_xy.bin', False)
            self.blocks_file_sink_13xy.set_unbuffered(False)
            self.blocks_file_sink_14xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_14_xy.bin', False)
            self.blocks_file_sink_14xy.set_unbuffered(False)
            self.blocks_file_sink_23xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_23_xy.bin', False)
            self.blocks_file_sink_23xy.set_unbuffered(False)
            self.blocks_file_sink_24xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_24_xy.bin', False)
            self.blocks_file_sink_24xy.set_unbuffered(False)
            self.blocks_file_sink_34xy = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_34_xy.bin', False)
            self.blocks_file_sink_34xy.set_unbuffered(False)
            self.blocks_file_sink_12yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_12_yx.bin', False)
            self.blocks_file_sink_12yx.set_unbuffered(False)
            self.blocks_file_sink_13yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_13_yx.bin', False)
            self.blocks_file_sink_13yx.set_unbuffered(False)
            self.blocks_file_sink_14yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_14_yx.bin', False)
            self.blocks_file_sink_14yx.set_unbuffered(False)
            self.blocks_file_sink_23yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_23_yx.bin', False)
            self.blocks_file_sink_23yx.set_unbuffered(False)
            self.blocks_file_sink_24yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_24_yx.bin', False)
            self.blocks_file_sink_24yx.set_unbuffered(False)
            self.blocks_file_sink_34yx = blocks.file_sink(gr.sizeof_gr_complex*lfft, rootname+'_34_yx.bin', False)
            self.blocks_file_sink_34yx.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_integrate_xx_12xx, 0), (self.blocks_file_sink_12xx, 0))
        self.connect((self.blocks_integrate_xx_13xx, 0), (self.blocks_file_sink_13xx, 0))
        self.connect((self.blocks_integrate_xx_14xx, 0), (self.blocks_file_sink_14xx, 0))
        self.connect((self.blocks_integrate_xx_23xx, 0), (self.blocks_file_sink_23xx, 0))
        self.connect((self.blocks_integrate_xx_24xx, 0), (self.blocks_file_sink_24xx, 0))
        self.connect((self.blocks_integrate_xx_34xx, 0), (self.blocks_file_sink_34xx, 0))
        self.connect((self.blocks_integrate_xx_12yy, 0), (self.blocks_file_sink_12yy, 0))
        self.connect((self.blocks_integrate_xx_13yy, 0), (self.blocks_file_sink_13yy, 0))
        self.connect((self.blocks_integrate_xx_14yy, 0), (self.blocks_file_sink_14yy, 0))
        self.connect((self.blocks_integrate_xx_23yy, 0), (self.blocks_file_sink_23yy, 0))
        self.connect((self.blocks_integrate_xx_24yy, 0), (self.blocks_file_sink_24yy, 0))
        self.connect((self.blocks_integrate_xx_34yy, 0), (self.blocks_file_sink_34yy, 0))
        if(npol>2):
            self.connect((self.blocks_integrate_xx_12xy, 0), (self.blocks_file_sink_12xy, 0))
            self.connect((self.blocks_integrate_xx_13xy, 0), (self.blocks_file_sink_13xy, 0))
            self.connect((self.blocks_integrate_xx_14xy, 0), (self.blocks_file_sink_14xy, 0))
            self.connect((self.blocks_integrate_xx_23xy, 0), (self.blocks_file_sink_23xy, 0))
            self.connect((self.blocks_integrate_xx_24xy, 0), (self.blocks_file_sink_24xy, 0))
            self.connect((self.blocks_integrate_xx_34xy, 0), (self.blocks_file_sink_34xy, 0))
            self.connect((self.blocks_integrate_xx_12yx, 0), (self.blocks_file_sink_12yx, 0))
            self.connect((self.blocks_integrate_xx_13yx, 0), (self.blocks_file_sink_13yx, 0))
            self.connect((self.blocks_integrate_xx_14yx, 0), (self.blocks_file_sink_14yx, 0))
            self.connect((self.blocks_integrate_xx_23yx, 0), (self.blocks_file_sink_23yx, 0))
            self.connect((self.blocks_integrate_xx_24yx, 0), (self.blocks_file_sink_24yx, 0))
            self.connect((self.blocks_integrate_xx_34yx, 0), (self.blocks_file_sink_34yx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_12xx, 0), (self.blocks_integrate_xx_12xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_13xx, 0), (self.blocks_integrate_xx_13xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_14xx, 0), (self.blocks_integrate_xx_14xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_23xx, 0), (self.blocks_integrate_xx_23xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_24xx, 0), (self.blocks_integrate_xx_24xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_34xx, 0), (self.blocks_integrate_xx_34xx, 0))
        self.connect((self.blocks_multiply_conjugate_cc_12yy, 0), (self.blocks_integrate_xx_12yy, 0))
        self.connect((self.blocks_multiply_conjugate_cc_13yy, 0), (self.blocks_integrate_xx_13yy, 0))
        self.connect((self.blocks_multiply_conjugate_cc_14yy, 0), (self.blocks_integrate_xx_14yy, 0))
        self.connect((self.blocks_multiply_conjugate_cc_23yy, 0), (self.blocks_integrate_xx_23yy, 0))
        self.connect((self.blocks_multiply_conjugate_cc_24yy, 0), (self.blocks_integrate_xx_24yy, 0))
        self.connect((self.blocks_multiply_conjugate_cc_34yy, 0), (self.blocks_integrate_xx_34yy, 0))
        if(npol>2):
            self.connect((self.blocks_multiply_conjugate_cc_12xy, 0), (self.blocks_integrate_xx_12xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_13xy, 0), (self.blocks_integrate_xx_13xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_14xy, 0), (self.blocks_integrate_xx_14xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_23xy, 0), (self.blocks_integrate_xx_23xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_24xy, 0), (self.blocks_integrate_xx_24xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_34xy, 0), (self.blocks_integrate_xx_34xy, 0))
            self.connect((self.blocks_multiply_conjugate_cc_12yx, 0), (self.blocks_integrate_xx_12yx, 0))
            self.connect((self.blocks_multiply_conjugate_cc_13yx, 0), (self.blocks_integrate_xx_13yx, 0))
            self.connect((self.blocks_multiply_conjugate_cc_14yx, 0), (self.blocks_integrate_xx_14yx, 0))
            self.connect((self.blocks_multiply_conjugate_cc_23yx, 0), (self.blocks_integrate_xx_23yx, 0))
            self.connect((self.blocks_multiply_conjugate_cc_24yx, 0), (self.blocks_integrate_xx_24yx, 0))
            self.connect((self.blocks_multiply_conjugate_cc_34yx, 0), (self.blocks_integrate_xx_34yx, 0))
        self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_12xx, 0))
        self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_12xx, 1))
        self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_13xx, 0))
        self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_13xx, 1))
        self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_14xx, 0))
        self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_14xx, 1))
        self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_23xx, 0))
        self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_23xx, 1))
        self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_24xx, 0))
        self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_24xx, 1))
        self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_34xx, 0))
        self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_34xx, 1))
        self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_12yy, 0))
        self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_12yy, 1))
        self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_13yy, 0))
        self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_13yy, 1))
        self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_14yy, 0))
        self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_14yy, 1))
        self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_23yy, 0))
        self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_23yy, 1))
        self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_24yy, 0))
        self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_24yy, 1))
        self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_34yy, 0))
        self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_34yy, 1))
        if(npol>2):
            self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_12xy, 0))
            self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_12xy, 1))
            self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_13xy, 0))
            self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_13xy, 1))
            self.connect((self.fft_vxx_1x, 0), (self.blocks_multiply_conjugate_cc_14xy, 0))
            self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_14xy, 1))
            self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_23xy, 0))
            self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_23xy, 1))
            self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_24xy, 0))
            self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_24xy, 1))
            self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_34xy, 0))
            self.connect((self.fft_vxx_4y, 0), (self.blocks_multiply_conjugate_cc_34xy, 1))
            self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_12yx, 0))
            self.connect((self.fft_vxx_2x, 0), (self.blocks_multiply_conjugate_cc_12yx, 1))
            self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_13yx, 0))
            self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_13yx, 1))
            self.connect((self.fft_vxx_1y, 0), (self.blocks_multiply_conjugate_cc_14yx, 0))
            self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_14yx, 1))
            self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_23yx, 0))
            self.connect((self.fft_vxx_3x, 0), (self.blocks_multiply_conjugate_cc_23yx, 1))
            self.connect((self.fft_vxx_2y, 0), (self.blocks_multiply_conjugate_cc_24yx, 0))
            self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_24yx, 1))
            self.connect((self.fft_vxx_3y, 0), (self.blocks_multiply_conjugate_cc_34yx, 0))
            self.connect((self.fft_vxx_4x, 0), (self.blocks_multiply_conjugate_cc_34yx, 1))
        self.connect((self.blocks_stream_to_vector_1x, 0), (self.fft_vxx_1x, 0))
        self.connect((self.blocks_stream_to_vector_1y, 0), (self.fft_vxx_1y, 0))
        self.connect((self.blocks_stream_to_vector_2x, 0), (self.fft_vxx_2x, 0))
        self.connect((self.blocks_stream_to_vector_2y, 0), (self.fft_vxx_2y, 0))
        self.connect((self.blocks_stream_to_vector_3x, 0), (self.fft_vxx_3x, 0))
        self.connect((self.blocks_stream_to_vector_3y, 0), (self.fft_vxx_3y, 0))
        self.connect((self.blocks_stream_to_vector_4x, 0), (self.fft_vxx_4x, 0))
        self.connect((self.blocks_stream_to_vector_4y, 0), (self.fft_vxx_4y, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_stream_to_vector_1x, 0))
        self.connect((self.uhd_usrp_source_0, 1), (self.blocks_stream_to_vector_1y, 0))
        self.connect((self.uhd_usrp_source_0, 2), (self.blocks_stream_to_vector_2x, 0))
        self.connect((self.uhd_usrp_source_0, 3), (self.blocks_stream_to_vector_2y, 0))
        self.connect((self.uhd_usrp_source_0, 4), (self.blocks_stream_to_vector_3x, 0))
        self.connect((self.uhd_usrp_source_0, 5), (self.blocks_stream_to_vector_3y, 0))
        self.connect((self.uhd_usrp_source_0, 6), (self.blocks_stream_to_vector_4x, 0))
        self.connect((self.uhd_usrp_source_0, 7), (self.blocks_stream_to_vector_4y, 0))

    def get_tint(self):
        return self.tint

    def set_tint(self, tint):
        self.tint = tint
        self.set_nav(int(self.tint*self.samp_rate/self.lfft))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_nav(int(self.tint*self.samp_rate/self.lfft))
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_lfft(self):
        return self.lfft

    def get_rootname(self):
        return self.rootname

    def get_npol(self):
        return self.npol

    def set_lfft(self, lfft):
        self.lfft = lfft
        self.set_nav(int(self.tint*self.samp_rate/self.lfft))

    def get_nav(self):
        return self.nav

    def set_nav(self, nav):
        self.nav = nav

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

        self.uhd_usrp_source_0.set_gain(self.gain, 1)

        self.uhd_usrp_source_0.set_gain(self.gain, 2)

        self.uhd_usrp_source_0.set_gain(self.gain, 3)

        self.uhd_usrp_source_0.set_gain(self.gain, 4)

        self.uhd_usrp_source_0.set_gain(self.gain, 5)

        self.uhd_usrp_source_0.set_gain(self.gain, 6)

        self.uhd_usrp_source_0.set_gain(self.gain, 7)


    def get_freq0(self):
        return self.freq0

    def set_freq0(self, freq0):
        self.freq0 = freq0
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 1)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 2)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 3)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 4)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 5)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 6)
        self.uhd_usrp_source_0.set_center_freq(self.freq0, 7)


def main(top_block_cls=dlite_npol, options=None):

    tb = top_block_cls()
    tb.start()
    time.sleep(DURATION+1)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
