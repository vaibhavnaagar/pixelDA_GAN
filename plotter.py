import matplotlib.pyplot as plt
import multiprocessing as mp
import threading
from queue import Queue
import numpy as np

class Plotter():
    """docstring for Plotter."""
    def __init__(self, name, num_lines=1, legends=[], xlabel="", ylabel="", title=""):
        self.figure_name = name
        self.num_lines = num_lines
        self.legends = legends
        self.queue = Queue()
        # self.process = mp.Process(target=self.update_plot, args=())
        self.process = threading.Thread(target=self.update_plot)
        self.initialize_plot(xlabel, ylabel, title)
        self.data = {i : np.empty(shape=(0, 2)) for i in range(num_lines)}
        self.process.start()

    def initialize_plot(self, xlabel, ylabel, title):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines = []
        while len(self.legends) < self.num_lines:
            self.legends += [""]
        for n in range(self.num_lines):
            self.lines += self.ax.plot([],[], 'o-', label=self.legends[n])
        self.figure.suptitle(title)
        self.ax.legend()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        # self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def draw_plot(self):
        #Update data (with the new _and_ the old points)
        for i, line in enumerate(self.lines):
            line.set_data(self.data[i][:,0], self.data[i][:,1])
        # self.lines.set_xdata(self.xdata)
        # self.lines.set_ydata(self.ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.savefig(self.figure_name)

    def update_plot(self):
        while True:
            print("HAHA")
            datapoints = self.queue.get()
            print(self.queue.empty())
            for i,dp in enumerate(datapoints):
                self.data[i] = np.vstack((self.data[i], dp))
            self.draw_plot()
            self.queue.task_done()

    def __call__(self, *datapoints):
        assert len(datapoints) == self.num_lines
        self.queue.put(datapoints)

    def clean_up(self):
        self.process.terminate()


if __name__ == '__main__':
    p = Plotter("test.jpeg", 2, "steps", "val", "TEST")
    for x in np.arange(0,10,0.5):
        p((x, np.exp(-x**2)+10*np.exp(-(x-7)**2)), (x, x))

    # p.queue.close()
    # p.queue.join_thread()
    import time
    time.sleep(10)
    print(p.queue.empty())
    # while not p.queue.empty():
    #     pass
    p.queue.join()
    # print(p.queue.empty())
    p.clean_up()
