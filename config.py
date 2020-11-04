class EnvironmentSettings():
    def __init__(self):
        self.set_default()

    def set_default(self):
        data_dir = '/path/to/good/place/to/put/stuff'
        self.workspace_dir = '{}/psreg_ws/'.format(data_dir)
        self.pretrained_networks = '{}/models'.format(data_dir)
        self.kitti_dir = '/path/to/kitti/odometry'
        self.threedmatch_dir = '/path/to/threedmatch'