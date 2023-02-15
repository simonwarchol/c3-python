import pyc3.c3 as c3
# import c3
c3_instance = c3.c3()


def analyze(palette, color_term_limit=1):
    return c3_instance.analyze(palette, color_term_limit)


def color_name_distance_matrix(palette):
    return c3_instance.color_name_distance_matrix(palette)
#
# #
# if __name__ == '__main__':
#     color_list = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',
#                   '#c5b0d5']
#     print(analyze(color_list, color_term_limit=10))
#     csd = c3_instance.get_color_salience_dict()
#     test = ''
