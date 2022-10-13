# 05.04.22------------------------------------self.spacing----------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#
#
import plotly.express as px


def plot3d(data, lower_boundary=None, upper_boundary=None, what_to_color="z", annotation_for_that="z [m]", take=0.01, tick0=0, dtick=None):
    _dat = data.sample(int(take * len(data)))
    if lower_boundary is None:
        lower_boundary = [data.x.min(), data.y.min()]
    if upper_boundary is None:
        upper_boundary = [data.x.max(), data.y.max()]
    fig = px.scatter_3d(_dat, y="y", x="x", z="z", color=what_to_color,
                        labels=dict(x="x [m]", y="y [m]", z="z [m]", color=annotation_for_that),
                        # hover_data=["x", "y", "z", "coord_hash", "DTMAnalysis"],
                        title="Top-view of the quadindex grid: Some maybe missing due to sampling, only one dimension is cutoff correctly",
                        color_continuous_scale=px.colors.sequential.Viridis)
    # fig.update_xaxes(range=[lower_boundary[0], upper_boundary[0]], tick0=tick0, dtick=dtick, autorange=False)
    # fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[lower_boundary[1], upper_boundary[1]], tick0=tick0, dtick=dtick, autorange=False)
    # fig.update_
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_traces(marker_size=1)
    fig['layout']['scene']['aspectmode'] = "data"
    fig.show()
    # exit()

# alignment["marker_size"] = 0.1
# df2["marker_size"] = 2
# df = pandas.concat([alignment, df2], keys=["x", "y", "z", "horizontal_distance", "marker_size"])
#
# fig = px.scatter_3d(df, y="y", x="x", z="z", color="horizontal_distance",
#                     labels=dict(x="x [m]", y="y [m]", z="z [m]", horizontal_type="IFC Name"), size='marker_size')
# fig.update_yaxes(scaleanchor="x", scaleratio=1)
# # fig.update_traces(marker_size=1)
# fig['layout']['scene']['aspectmode'] = "data"
