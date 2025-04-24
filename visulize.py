from utils.setup import load_data
from utils.plot_scatter import plot_scatter

data = load_data()

plot_scatter(data['rho*'], data['D*'], 'rho*', 'D*')