import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

class Plotter():
    """docstring for Plotter."""
    def __init__(self, name, num_lines=1, legends=[], xlabel="", ylabel="", title=""):
        self.figure_name = name
        self.num_lines = num_lines
        self.legends = legends
        self.initialize_plot(xlabel, ylabel, title)
        self.data = {i : np.empty(shape=(0, 2)) for i in range(num_lines)}
        manager = mp.Manager()
        self.queue = manager.JoinableQueue()
        self.process = mp.Process(target=self.update_plot, args=(self.queue,))
        self.process.start()

    def initialize_plot(self, xlabel, ylabel, title):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines = []
        while len(self.legends) < self.num_lines:
            self.legends += [""]
        for n in range(self.num_lines):
            self.lines += self.ax.plot([],[], 'o-', label=self.legends[n], markersize=1)
        self.figure.suptitle(title)
        self.ax.legend()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        # self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.grid()

    def draw_plot(self):
        #Update data (with the new _and_ the old points)
        for i, line in enumerate(self.lines):
            line.set_data(self.data[i][:,0], self.data[i][:,1])
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.savefig(self.figure_name)

    def update_plot(self, q):
        while True:
            datapoints = q.get()
            for i,dp in enumerate(datapoints):
                self.data[i] = np.vstack((self.data[i], dp))
            self.draw_plot()
            q.task_done()

    def __call__(self, *datapoints):
        assert len(datapoints) == self.num_lines
        self.queue.put(datapoints)

    def clean_up(self):
        self.process.terminate()


if __name__ == '__main__':
    p = Plotter("test.jpeg", 2, "steps", "val", "TEST")
    for x in np.arange(0,100,0.5):
        print(x)
        p((x, np.exp(-x**2)+10*np.exp(-(x-7)**2)), (x, x))
    print(p.queue.empty())
    p.queue.join()
    print(p.queue.empty())
    p.clean_up()
