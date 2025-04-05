import plotly.graph_objects as go
import plotly.express as px


def visualisation(y_test, y_pred):
    """
    Assuming y_test and y_pred are 2D arrays
    """

    n_samples = 1  # number of samples to plot this is max i figure works

    trace_y_test = go.Scatter(
        x=list(range(len(y_test))),
        y=y_test,
        mode="lines",
        name="y_test",
        line=dict(color="blue"),
    )
    trace_y_pred = go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode="lines",
        name="y_pred",
        line=dict(color="red"),
    )

    steps = []

    fig = go.Figure(data=[trace_y_test, trace_y_pred])

    for i in range(n_samples):
        step = dict(
            method="update",
            args=[
                {
                    "y": [y_test[i], y_pred[i]],
                    "x": [list(range(len(y_test))), list(range(len(y_pred)))],
                },
                {"title": f"Sample {i}"},
            ],
            label=str(i),
        )
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Sample: "}, pad={"t": 50}, steps=steps)
    ]
    fig.update_layout(sliders=sliders)

    return fig
