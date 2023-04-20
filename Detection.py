
#
#
# def calculate_points(team_data):
#     points = 0
#     points += int(team_data[5])
#     points -= int(team_data[6])
#     points += 3 * int(team_data[5] > team_data[6])
#     points += int(team_data[5] == team_data[6])
#     return points
#
#
# for group, games in group_data.items():
#
#
#     team_data = {}
# with open("worldcupdata_4.csv", "r") as file:
#     for line in file:
#         game = line.strip().split()
#         if game[0] not in team_data:
#             team_data[game[0]] = [0, 0, 0, 0, 0]
#         if game[1] not in team_data:
#             team_data[game[1]] = [0, 0, 0, 0, 0]
#         for i in range(4, len(game)):
#             if game[i].isdigit():
#                 if i < 8:
#                     team_data[game[0]][i-4] += int(game[i])
#                 else:
#                     team_data[game[2]][i-12] += int(game[i])
# print(team_data)
#
#
# for game in games:
#
#
#     for i in range(4, 6):
#         country = game[i]
#     if country not in team_data:
#         team_data[country] = [0, 0, 0, 0, 0, 0, 0]
#
#
#         for i in range(4, 12):
#             team_data[game[i]][i - 4] += int(game[i + 8])
#
#
#     team_points = []
#     for country, data in team_data.items():
#         points = calculate_points(data)
#         team_points.append([country, points, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
#
#
#     team_points.sort(key=lambda x: (-x[1], -x[2], -x[3]))
#
#
#     print(group)
#     for i in range(2):
#         print(f"{i+1}. {team_points[i][0]} ({team_points[i][1]} points)")
#     print()