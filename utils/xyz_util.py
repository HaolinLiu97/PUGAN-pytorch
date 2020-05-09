
def save_xyz_file(numpy_array, xyz_dir):
    num_points = numpy_array.shape[0]
    with open(xyz_dir, 'w') as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (numpy_array[i, 0], numpy_array[i, 1], numpy_array[i, 2])
            f.write(line)
    return
