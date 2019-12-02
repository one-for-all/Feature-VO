"""Write data to file
"""

class RelativePoseWriter:
    def __init__(self, path):
        self.f = open(path, "w+")
    
    def write_pose(self, i, j, pose):
        # Write indices of frames for the relative pose
        self.f.write("{} {}".format(i, j))
        t, q = pose.to_t_q()
        values = t.tolist() + q.tolist()
        for v in values:
            self.f.write(" {}".format(v))
        self.f.write("\n")
