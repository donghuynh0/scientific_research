import plotly.express as px
import pandas as pd
import numpy as np


def plot_scatter(x, y, x_label=None, y_label=None, title=None):

    fig = px.scatter(
        x=x,
        y=y,
        labels={'x': x_label, 'y': y_label},
        title=title
    )

    fig.show()


def plot_relative_error(X_test, y_test, y_pred, target_name="D*"):
    df = pd.DataFrame({
        'T*': X_test['T*'],
        'rho*': X_test['rho*'],
        f'Actual {target_name}': np.round(y_test, 3),
        f'Predicted {target_name}': np.round(y_pred, 3)
    })

    df['Absolute Error'] = np.abs(df[f'Actual {target_name}'] - df[f'Predicted {target_name}'])
    df['Relative Error (%)'] = np.round((df['Absolute Error'] / df[f'Actual {target_name}']) * 100, 3)

    fig = px.scatter(
        df,
        x='rho*',
        y='Relative Error (%)',
        color='T*',
        title=f"Relative Error vs. ρ* (rho) with target = {target_name}",
        labels={'rho*': 'ρ* (rho)', 'Relative Error (%)': 'Relative Error (%)'},
        hover_data={
            'T*': True,
            f'Actual {target_name}': ':.3f',
            f'Predicted {target_name}': ':.3f',
            'Relative Error (%)': ':.3f'
        }
    )

    fig.update_traces(
        hovertemplate=f"<b>ρ*: </b> %{{x}}"+
                      f"<br><b>Relative Error: </b> %{{y:.3f}}%"+
                      f"<br><b>T*: </b> %{{customdata[0]}}"+
                      f"<br><b>Actual {target_name}: </b> %{{customdata[1]:.3f}}"+
                      f"<br><b>Predicted {target_name}: </b> %{{customdata[2]:.3f}}"
    )

    fig.show()
