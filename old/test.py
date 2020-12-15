from flask import Flask, render_template, make_response
from io import BytesIO
import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random
import numpy as np


app = Flask(__name__)

fig = plt.figure()
ax = fig.add_subplot()

x = np.arange(0,100,0.1)
y = x**2

@app.route('/')
def index():
  plt.cla()

  plt.title('Graph')
  plt.legend()
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.plot(x, y)
  

  canvas = FigureCanvasAgg(fig)
  png_output = BytesIO()
  canvas.print_png(png_output)
  data = png_output.getvalue()

  response = make_response(data)
  response.headers['Content-Type'] = 'image/png'
 

  return response


if __name__ == "__main__":
  app.run()
