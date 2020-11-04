import visdom
import visdom.server
import torch
import copy
import numpy as np

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

class VisBase:
    def __init__(self, visdom, show_data, title):
        self.visdom = visdom
        self.show_data = show_data
        self.title = title
        self.raw_data = None
        self.opts = {}

    def update(self, data, opts={}):
        self.save_data(data, opts)

        if self.show_data:
            self.draw_data()

    def save_data(self, data, opts):
        raise NotImplementedError

    def draw_data(self):
        raise NotImplementedError

    def toggle_display(self, new_mode=None):
        if new_mode is not None:
            self.show_data = new_mode
        else:
            self.show_data = not self.show_data

        if self.show_data:
            self.draw_data()
        else:
            self.visdom.close(self.title)


class VisImage(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        data = data.float()
        self.raw_data = data
        self.opts = opts

    def draw_data(self):
        self.visdom.image(self.raw_data.clone(), opts={'title': self.title}, win=self.title)


class VisHeatmap(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        data = data.squeeze().flip(0)
        self.raw_data = data

    def draw_data(self):
        self.visdom.heatmap(self.raw_data.clone(),  opts={'title': self.title}, win=self.title)


class VisCostVolume(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)
        self.show_slice = False
        self.slice_pos = None

    def show_cost_volume(self):
        data = self.raw_data.clone()

        # data_perm = data.permute(2, 0, 3, 1).contiguous()
        data_perm = data.permute(0, 2, 1, 3).contiguous()
        data_perm = data_perm.view(data_perm.shape[0] * data_perm.shape[1], -1)
        self.visdom.heatmap(data_perm.flip(0), opts={'title': self.title}, win=self.title)

    def set_zoom_pos(self, slice_pos):
        self.slice_pos = slice_pos

    def toggle_show_slice(self, new_mode=None):
        if new_mode is not None:
            self.show_slice = new_mode
        else:
            self.show_slice = not self.show_slice

    def show_cost_volume_slice(self):
        slice_pos = self.slice_pos

        # slice_pos: [row, col]
        cost_volume_data = self.raw_data.clone()

        cost_volume_slice = cost_volume_data[slice_pos[0], slice_pos[1], :, :]
        self.visdom.heatmap(cost_volume_slice.flip(0), opts={'title': self.title}, win=self.title)

    def save_data(self, data, opts):
        data = data.view(data.shape[-2], data.shape[-1], data.shape[-2], data.shape[-1])
        self.raw_data = data

    def draw_data(self):
        if self.show_slice:
            self.show_cost_volume_slice()
        else:
            self.show_cost_volume()


class VisCostVolumeUI(VisBase):
    def cv_ui_handler(self, data):
        zoom_toggled = False
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'ArrowRight':
                self.zoom_pos[1] = min(self.zoom_pos[1] + 1, self.feat_shape[1]-1)
            elif data['key'] == 'ArrowLeft':
                self.zoom_pos[1] = max(self.zoom_pos[1] - 1, 0)
            elif data['key'] == 'ArrowUp':
                self.zoom_pos[0] = max(self.zoom_pos[0] - 1, 0)
            elif data['key'] == 'ArrowDown':
                self.zoom_pos[0] = min(self.zoom_pos[0] + 1, self.feat_shape[0]-1)
            elif data['key'] == 'Enter':
                self.zoom_mode = not self.zoom_mode
                zoom_toggled = True

        # Update image
        self.show_image()

        # Update cost volumes
        for block_title, block in self.registered_blocks.items():
            if isinstance(block, VisCostVolume):
                block.set_zoom_pos(self.zoom_pos)
                block.toggle_show_slice(self.zoom_mode)

                if (self.zoom_mode or zoom_toggled) and block.show_data:
                    block.draw_data()

    def __init__(self, visdom, show_data, title, feat_shape, registered_blocks):
        super().__init__(visdom, show_data, title)
        self.feat_shape = feat_shape
        self.zoom_mode = False
        self.zoom_pos = [int((feat_shape[0] - 1) / 2), int((feat_shape[1] - 1) / 2)]
        self.registered_blocks = registered_blocks

        self.visdom.register_event_handler(self.cv_ui_handler, title)

    def draw_grid(self, data):
        stride_r = int(data.shape[1] / self.feat_shape[0])
        stride_c = int(data.shape[2] / self.feat_shape[1])

        # Draw grid
        data[:, list(range(0, data.shape[1], stride_r)), :] = 0
        data[:, :, list(range(0, data.shape[2], stride_c))] = 0

        data[0, list(range(0, data.shape[1], stride_r)), :] = 255
        data[0, :, list(range(0, data.shape[2], stride_c))] = 255

        return data

    def shade_cell(self, data):
        stride_r = int(data.shape[1] / self.feat_shape[0])
        stride_c = int(data.shape[2] / self.feat_shape[1])

        r1 = self.zoom_pos[0]*stride_r
        r2 = min((self.zoom_pos[0] + 1)*stride_r, data.shape[1])

        c1 = self.zoom_pos[1] * stride_c
        c2 = min((self.zoom_pos[1] + 1) * stride_c, data.shape[2])

        factor = 0.8 if self.zoom_mode else 0.5
        data[:, r1:r2, c1:c2] = data[:, r1:r2, c1:c2] * (1 - factor) + torch.tensor([255.0, 0.0, 0.0]).view(3, 1, 1) * factor
        return data

    def show_image(self, data=None):
        if data is None:
            data = self.raw_data.clone()

        data = self.draw_grid(data)
        data = self.shade_cell(data)
        self.visdom.image(data, opts={'title': self.title}, win=self.title)

    def save_data(self, data, opts):
        # Ignore feat shape
        data = data[0]
        data = data.float()
        self.raw_data = data

    def draw_data(self):
        self.show_image(self.raw_data.clone())


class VisInfoDict(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def generate_display_text(self, data):
        display_text = ''
        for key, value in data.items():
            key = key.replace('_', ' ')
            if value is None:
                display_text += '<b>{}</b>: {}<br>'.format(key, 'None')
            elif isinstance(value, (str, int)):
                display_text += '<b>{}</b>: {}<br>'.format(key, value)
            else:
                display_text += '<b>{}</b>: {:.2f}<br>'.format(key, value)

        return display_text

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        display_text = self.generate_display_text(data)
        self.visdom.text(display_text, opts={'title': self.title}, win=self.title)


class VisText(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        self.visdom.text(data, opts={'title': self.title}, win=self.title)


class VisLinePlot(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        if isinstance(self.raw_data, (list, tuple)):
            data_y = self.raw_data[0].clone()
            data_x = self.raw_data[1].clone()
        else:
            data_y = self.raw_data.clone()
            data_x = torch.arange(data_y.shape[0])

        self.visdom.line(data_y, data_x, opts={'title': self.title}, win=self.title)


class VisPointClouds(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        pcds = self.raw_data["pcds"]
        X = self.raw_data["X"]
        c = self.raw_data["c"]

        pc_list = [p.permute(1,0) for p in pcds]
        M = len(pc_list)
        pcds_all = torch.cat(pc_list)

        if c is None:
            c = []
            for idx in range(M):
                c.append(torch.zeros((pcds[idx].shape[-1],), dtype=torch.int8) + idx)

            c = torch.cat(c).contiguous()

        if not X is None:
            Xs = X.squeeze()
            pcds_all = torch.cat([pcds_all, Xs])
            c = torch.cat([c, torch.zeros((Xs.shape[0],), dtype=torch.int8) + M])

        self.visdom.scatter(X=pcds_all, Y=c + 1, opts=dict(title=self.title, markersize=2), win=self.title)

class Visdom:
    def __init__(self, debug=0, ui_info=None, visdom_info=None):
        self.debug = debug
        self.visdom = visdom.Visdom(server=visdom_info.get('server', '127.0.0.1'), port=visdom_info.get('port', 8097))
        self.registered_blocks = {}
        self.blocks_list = []

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')
        self.visdom.register_event_handler(self.block_list_callback_handler, 'block_list')

        if ui_info is not None:
            self.visdom.register_event_handler(ui_info['handler'], ui_info['win_id'])

    def block_list_callback_handler(self, data):
        field_name = self.blocks_list[data['propertyId']]['name']

        self.registered_blocks[field_name].toggle_display(data['value'])

        self.blocks_list[data['propertyId']]['value'] = data['value']

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

    def register(self, data, mode, debug_level=0, title='Data', opts={}):
        if title not in self.registered_blocks.keys():
            show_data = self.debug >= debug_level

            if title is not 'Tracking':
                self.blocks_list.append({'type': 'checkbox', 'name': title, 'value': show_data})

            self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

            if mode == 'image':
                self.registered_blocks[title] = VisImage(self.visdom, show_data, title)
            elif mode == 'heatmap':
                self.registered_blocks[title] = VisHeatmap(self.visdom, show_data, title)
            elif mode == 'cost_volume':
                self.registered_blocks[title] = VisCostVolume(self.visdom, show_data, title)
            elif mode == 'cost_volume_ui':
                self.registered_blocks[title] = VisCostVolumeUI(self.visdom, show_data, title, data[1],
                                                                self.registered_blocks)
            elif mode == 'info_dict':
                self.registered_blocks[title] = VisInfoDict(self.visdom, show_data, title)
            elif mode == 'text':
                self.registered_blocks[title] = VisText(self.visdom, show_data, title)
            elif mode == 'lineplot':
                self.registered_blocks[title] = VisLinePlot(self.visdom, show_data, title)
            elif mode == 'point_clouds':
                self.registered_blocks[title] = VisPointClouds(self.visdom, show_data, title)
            else:
                raise ValueError('Visdom Error: Unknown data mode {}'.format(mode))
        # Update
        self.registered_blocks[title].update(data, opts)

