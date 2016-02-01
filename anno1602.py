#!/usr/bin/env python
# coding: utf-8

from getpass import getuser

# Settings of the annotation process
############

# How many frames does a batch scan.
# This excludes the very last frame, since we start at 0.
batchsize = 100
# Show every tickskip-th frame for annotation.
tickskip = 5
# Skip so many batches after each. Example: 3 means batch, skip, skip, skip, batch, ...
batchskip = 3

# Everything further away than this is dropped.
# This is because many lasers put NaN to some number, e.g. 29.999
laser_cutoff = 14

# Where to load the laser csv data and images from.
# TODO: Describe better!
basedir = "/work/" + getuser() + "/strands/wheelchair/dumped/"
# Where to save the detections to.
# TODO: Describe better!
savedir = "/media/" + getuser() + "/NSA1/strands/wheelchair/"

# The field-of-view of the laser you're using.
# From https://github.com/lucasb-eyer/strands_karl/blob/5a2dd60/launch/karl_robot.launch#L25
# TODO: change to min/max for supporting non-zero-centered lasers.
laserFoV = 225

# The field-of-view of the supportive camera you're using.
# From https://www.asus.com/3D-Sensor/Xtion_PRO_LIVE/specifications/
# TODO: change to min/max for supporting non-zero-centered cameras.
cameraFoV = 58

# The size of the camera is needed for pre-generating the image-axes in the plot for efficiency.
camsize = (480, 640)

# Radius of the circle around the cursor, in data-units.
# From https://thesegamesiplay.files.wordpress.com/2015/03/wheelchair.jpg
circrad = 1.22/2

# TODO: make the types of labels configurable? Although, does it even matter?

# End of settings
############

import sys
import os
import json
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, AxesWidget

try:
    from cv2 import imread
except ImportError:
    from matplotlib.image import imread

laserFoV = np.radians(laserFoV)
cameraFoV = np.radians(cameraFoV)

if len(sys.argv) < 2:
    print("Usage: {} relative/path/to_file.bag".format(sys.argv[0]))
    print()
    print("relative to {}".format(basedir))
    sys.exit(1)

name = sys.argv[1]


def mkdirs(fname):
    """ Make directories necessary for creating `fname`. """
    dname = os.path.dirname(fname)
    if not os.path.isdir(dname):
        os.makedirs(dname)


# **TODO**: put into common toolbox repo.
def xy_to_rphi(x, y):
    # Note: axes rotated by 90 by intent, so that 0 is top.
    return np.hypot(x, y), np.arctan2(-x, y)


def rphi_to_xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan_to_xy(scan, thresh=None):
    s = np.array(scan, copy=True)
    if thresh is not None:
        s[s > thresh] = np.nan
    angles = np.linspace(-laserFoV/2, laserFoV/2, len(scan))
    return rphi_to_xy(scan, angles)


def imload(name, *seqs):
    for s in seqs:
        fname = "{}{}_dir/{}.jpg".format(basedir, name, int(s))
        im = imread(fname)
        if im is not None:
            return im[:,:,::-1]
    print("WARNING: Couldn't find any of " + ' ; '.join(map(str, map(int, seqs))))
    return np.zeros(camsize + (3,), dtype=np.uint8)


class Anno1602:
    def __init__(self, batches, scans, seqs, laser_thresh=laser_cutoff, circrad=circrad, xlim=None, ylim=None):
        self.batches = batches
        self.scans = scans
        self.seqs = seqs
        self.b = 0
        self.i = 0
        self.circrad = circrad
        self.xlim = xlim
        self.ylim = ylim

        # Build the figure and the axes.
        self.fig = plt.figure(figsize=(10,10))
        gs = mpl.gridspec.GridSpec(3, 2, width_ratios=(3, 1))

        self.axlaser = plt.subplot(gs[:,0])
        axprev, axpres, axnext = plt.subplot(gs[0,1]), plt.subplot(gs[1,1]), plt.subplot(gs[2,1])

        self.imprev = axprev.imshow(np.random.randint(255, size=camsize + (3,)), interpolation='nearest', animated=True)
        self.impres = axpres.imshow(np.random.randint(255, size=camsize + (3,)), interpolation='nearest', animated=True)
        self.imnext = axnext.imshow(np.random.randint(255, size=camsize + (3,)), interpolation='nearest', animated=True)
        axprev.axis('off')
        axpres.axis('off')
        axnext.axis('off')

        self.circ = MouseCircle(self.axlaser, radius=self.circrad, linewidth=1, fill=False)

        # Configure interaction
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.key)

        # Labels!!
        self.wheelchairs = [[None for i in b] for b in batches]
        self.walkingaids = [[None for i in b] for b in batches]
        self.load()

        self.replot()

    def save(self):
        mkdirs(savedir + name)

        def _doit(f, data):
            for ib, batch in enumerate(data):
              if batch is not None:
                for i, seq in enumerate(batch):
                  if seq is not None:
                    f.write('{},['.format(self.seqs[self.batches[ib][i]]))
                    f.write(','.join('[{},{}]'.format(*xy_to_rphi(x,y)) for x,y in seq))
                    f.write(']\n')

        with open(savedir + name + ".wc", "w+") as f:
            _doit(f, self.wheelchairs)

        with open(savedir + name + ".wa", "w+") as f:
            _doit(f, self.walkingaids)

    def load(self):
        def _doit(f, whereto):
            # loading a file can be done here and should "just" be reading it
            # line-by-line, de-json-ing the second half of `,` and recording it in
            # a dict with the sequence number which is the first half of `,`.
            data = {}
            for line in f:
                seq, tail = line.split(',', 1)
                data[int(seq)] = [rphi_to_xy(r, phi) for r,phi in json.loads(tail)]

            # Then, in second pass, go through b/i and check for `seqs[batches[b][i]]`
            # in that dict, and use that.
            for ib, batch in enumerate(whereto):
                for i, _ in enumerate(batch):
                    batch[i] = data.get(self.seqs[self.batches[ib][i]], None)

        try:
            with open(savedir + name + ".wc", "r") as f:
                _doit(f, self.wheelchairs)

            with open(savedir + name + ".wa", "r") as f:
                _doit(f, self.walkingaids)
        except FileNotFoundError:
            pass  # That's ok, just means no annotations yet.

    def replot(self, newbatch=True):
        batch = self.batches[self.b]

        self.axlaser.clear()
        self.axlaser.scatter(*scan_to_xy(self.scans[batch[self.i]], self.laser_thresh), s=10, color='#E24A33', alpha=0.5, lw=0)

        # Camera frustum to help orientation.
        self.axlaser.plot([0, -self.laser_thresh*np.sin(cameraFoV/2)], [0, self.laser_thresh*np.cos(cameraFoV/2)], 'k:')
        self.axlaser.plot([0,  self.laser_thresh*np.sin(cameraFoV/2)], [0, self.laser_thresh*np.cos(cameraFoV/2)], 'k:')

        for x,y in self.wheelchairs[self.b][self.i] or []:
            self.axlaser.scatter(x, y, marker='+', s=50, color='#348ABD')
        for x,y in self.walkingaids[self.b][self.i] or []:
            self.axlaser.scatter(x, y, marker='x', s=50, color='#988ED5')

        # Fix aspect ratio and visible region.
        if self.xlim is not None:
            self.axlaser.set_xlim(*self.xlim)
        if self.ylim is not None:
            self.axlaser.set_ylim(*self.ylim)
        self.axlaser.set_aspect('equal', adjustable='box')  # Force axes to have equal scale.

        b = self.seqs[batch[self.i]]
        self.impres.set_data(imload(name, b, b-1, b+1, b-2, b+2))

        if newbatch:
            a = self.seqs[batch[0]]
            self.imprev.set_data(imload(name, a, a+1, a+2, a+tickskip, a+batchsize//10, a+batchsize//5, a+batchsize//4))
            c = self.seqs[batch[-1]]
            self.imnext.set_data(imload(name, c, c-1, c-2, c-tickskip, c-batchsize//10, c-batchsize//5, c-batchsize//4))

        self.fig.suptitle("{}: Batch {}/{} frame {}, seq {}".format(name, self.b+1, len(self.batches), self.i*tickskip, self.seqs[batch[self.i]]))
        self.fig.canvas.draw()
        self.circ._update()

    def click(self, e):
        if self._ignore(e):
            return

        if self.wheelchairs[self.b][self.i] is None:
            self.wheelchairs[self.b][self.i] = []
        if self.walkingaids[self.b][self.i] is None:
            self.walkingaids[self.b][self.i] = []

        if e.button == 1:
            self.wheelchairs[self.b][self.i].append((e.xdata, e.ydata))
        elif e.button == 3:
            self.walkingaids[self.b][self.i].append((e.xdata, e.ydata))
        elif e.button == 2:
            self._clear(e.xdata, e.ydata)

        self.replot(newbatch=False)

    def scroll(self, e):
        if self._ignore(e):
            return

        if e.button == 'down':
            for _ in range(int(-e.step)):
                self._nexti()
        elif e.button == 'up':
            for _ in range(int(e.step)):
                self._previ()
        self.replot(newbatch=False)

    def key(self, e):
        if self._ignore(e):
            return

        newbatch = False
        if e.key in ("left", "a"):
            self._nexti()
        elif e.key in ("right", "d"):
            self._previ()
        elif e.key in ("down", "s", "pagedown"):
            self._prevb()
            newbatch = True
        elif e.key in ("up", "w", "pageup"):
            self._nextb()
            newbatch = True
        elif e.key == "c":
            self._clear(e.xdata, e.ydata)
        else:
            print(e.key)

        self.replot(newbatch=newbatch)

    def _nexti(self):
        self.i = max(0, self.i-1)

    def _previ(self):
        self.i = min(len(self.batches[self.b])-1, self.i+1)

    def _nextb(self):
        self.save()
        self.b += 1
        self.i = 0

        # We decided to close after the last batch.
        if self.b == len(self.batches):
            self.b -= 1  # Because it gets redrawn once before closed...
            plt.close(self.fig)

    def _prevb(self):
        self.save()
        self.b = max(0, self.b-1)
        self.i = 0

    def _clear(self, mx, my):
      try:
        self.wheelchairs[self.b][self.i] = [(x,y) for x,y in self.wheelchairs[self.b][self.i] if np.hypot(mx-x, my-y) > self.circrad]
        self.walkingaids[self.b][self.i] = [(x,y) for x,y in self.walkingaids[self.b][self.i] if np.hypot(mx-x, my-y) > self.circrad]
      except TypeError:
          import pdb ; pdb.set_trace() # THERE IS A RARE BUG HERE. CALL LUCAS

    def _ignore(self, e):
        # https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Interactive_Plotting.html#Handling_click_events_while_zoomed
        # But we don't do `not e.inaxes` because that one is only set when the mouse moves, meaning
        # when we switch the frame, it will not be `inaxes` as long as we don't move the mouse!
        return plt.get_current_fig_manager().toolbar.mode != ''


# Annoyingly large amount of code for just a circle around the mouse cursor!
class MouseCircle(AxesWidget):
    def __init__(self, ax, radius, **circprops):
        AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.storebg)

        circprops['animated'] = True
        circprops['radius'] = radius

        self.circ = plt.Circle((0,0), **circprops)
        self.ax.add_artist(self.circ)

        self.background = None

    def storebg(self, e):
        if not self.ignore(e):
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def onmove(self, e):
        if not self.ignore(e) and self.canvas.widgetlock.available(self):
            self.circ.center = (e.xdata, e.ydata)  # (None,None) -> invisible if out of axis.
            self._update()

    def _update(self):
        if self.background is not None:
           self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.circ)
        self.canvas.blit()  # Note: not passing `self.ax.bbox` as else the other axes stay old...


if __name__ == "__main__":
    # Loading all data first
    print("Loading data...") ; sys.stdout.flush()
    data = np.genfromtxt(basedir + name + ".csv", delimiter=",")
    seqs, scans = data[:,0].astype(np.uint32), data[:,1:-1]  # Last column is always empty for ease of dumping.
    print("Loaded {} scans".format(len(seqs))) ; sys.stdout.flush()

    # Chunking into minibatches.
    print("Chunking into batches...") ; sys.stdout.flush()
    batches = []
    for bstart in np.arange(tickskip, len(seqs)-tickskip, batchsize*(batchskip+1)):
        batches.append(np.arange(bstart, bstart+batchsize-1-tickskip, tickskip))
    print("Created {} batches".format(len(batches))) ; sys.stdout.flush()

    # Determine the view-space.
    xr, yr = scan_to_xy(np.full(scans.shape[1], laser_cutoff, dtype=np.float32))

    print("Starting annotator...") ; sys.stdout.flush()
    anno = Anno1602(batches, scans, seqs, laser_cutoff, xlim=(min(xr), max(xr)), ylim=(min(yr), max(yr)))

    t0 = time.time()
    anno.replot()
    plt.show()
    t1 = time.time()

    print("You annotated {} batches in {:.0f}s, i.e. took {:.1f}s per batch.".format(len(batches), t1-t0, (t1-t0)/len(batches)))
