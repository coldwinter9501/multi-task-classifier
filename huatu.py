import matplotlib.pyplot as plt

name_list = ['mar', 'net', 'pea','pea+sph', 'pea+wid','sph','sph+wid']
num_list = [36, 212, 124, 107, 27, 374, 81]
rects = plt.bar(range(len(num_list)), num_list, color='r')

index = [0, 1, 2, 3, 4, 5, 6]
index = [float(c) + 0.4 for c in index]
plt.xticks(index, name_list)
plt.ylabel("count ")
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
plt.savefig('distribution ')
plt.show()