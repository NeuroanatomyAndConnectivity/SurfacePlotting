class PlotSurfaces(object):

    from nilearn._utils.compat import _basestring

    # Import libraries
    import numpy as np
    import nibabel
    from nibabel import gifti
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from mpl_toolkits.mplot3d import Axes3D

    def __init__(self,
                 meshes,
                 backgrounds,
                 labels = None,
                 cmap = 'jet',
                 dmin = 0,
                 dmax = 10):

        # Surface for all plots
        self.meshes = meshes
        self.backgrounds = backgrounds
        self.labels = labels

        # Color map for all plots
        self.cmap = cmap

        # Min and max for the color scale
        self.dmin = dmin
        self.dmax = dmax

        self.coords_ = {}
        self.faces_ = {}

        self.coords_['lh'], self.faces_['lh'] = self.check_surf_mesh(0)
        self.coords_['rh'], self.faces_['rh'] = self.check_surf_mesh(1)

        self.bgs_ = {}
        self.bgs_['lh'] = self.check_surf_data(0)
        self.bgs_['rh'] = self.check_surf_data(1)

        self.cortex_ = {}
        if self.labels is not None:
            self.cortex_['lh'] = self.get_cortical_indices(0)
            self.cortex_['rh'] = self.get_cortical_indices(1)

        self.plots_ = {}
        self.hind_ = {}


    def _get_plot_stat_map_params(self, stat_map_data, vmax, symmetric_cbar, kwargs,
        force_min_stat_map_value=None):
        """ Internal function for setting value limits for plot_stat_map and
        plot_glass_brain.
        The limits for the colormap will always be set to range from -vmax to vmax.
        The limits for the colorbar depend on the symmetric_cbar argument, please
        refer to docstring of plot_stat_map.
        """
        # make sure that the color range is symmetrical
        if vmax is None or symmetric_cbar in ['auto', False]:
            # Avoid dealing with masked_array:
            if hasattr(stat_map_data, '_mask'):
                stat_map_data = np.asarray(
                        stat_map_data[np.logical_not(stat_map_data._mask)])
            stat_map_max = np.nanmax(stat_map_data)
            if force_min_stat_map_value == None:
                stat_map_min = np.nanmin(stat_map_data)
            else:
                stat_map_min = force_min_stat_map_value
        if symmetric_cbar == 'auto':
            symmetric_cbar = stat_map_min < 0 and stat_map_max > 0
        if vmax is None:
            vmax = max(-stat_map_min, stat_map_max)
        if 'vmin' in kwargs:
            raise ValueError('this function does not accept a "vmin" '
                'argument, as it uses a symmetrical range '
                'defined via the vmax argument. To threshold '
                'the map, use the "threshold" argument')
        vmin = -vmax
        if not symmetric_cbar:
            negative_range = stat_map_max <= 0
            positive_range = stat_map_min >= 0
            if positive_range:
                cbar_vmin = 0
                cbar_vmax = None
            elif negative_range:
                cbar_vmax = 0
                cbar_vmin = None
            else:
                cbar_vmin = stat_map_min
                cbar_vmax = stat_map_max
        else:
            cbar_vmin, cbar_vmax = None, None
        return cbar_vmin, cbar_vmax, vmin, vmax

    # function to figure out datatype and load data
    def check_surf_data(self, index, gii_darray=0):

        from nilearn._utils.compat import _basestring
        import nibabel

        surf_data = self.backgrounds[index]

        # if the input is a filename, load it
        if isinstance(surf_data, _basestring):
            if (surf_data.endswith('nii') or surf_data.endswith('nii.gz') or
                    surf_data.endswith('mgz')):
                data = np.squeeze(nibabel.load(surf_data).get_data())
            elif (surf_data.endswith('curv') or surf_data.endswith('sulc') or
                    surf_data.endswith('thickness')):
                data = nibabel.freesurfer.io.read_morph_data(surf_data)
            elif surf_data.endswith('annot'):
                data = nibabel.freesurfer.io.read_annot(surf_data)[0]
            elif surf_data.endswith('label'):
                data = nibabel.freesurfer.io.read_label(surf_data)
            elif surf_data.endswith('gii'):
                data = gifti.read(surf_data).darrays[gii_darray].data
            else:
                raise ValueError('Format of data file not recognized.')
        # if the input is an array, it should have a single dimension
        elif isinstance(surf_data, np.ndarray):
            data = np.squeeze(surf_data)
            if len(data.shape) is not 1:
                raise ValueError('Data array cannot have more than one dimension.')
        return data

    # function to figure out datatype and load data
    def check_surf_mesh(self, index):

        from nilearn._utils.compat import _basestring
        import nibabel

        # if input is a filename, try to load it

        surf_mesh = self.meshes[index]

        if isinstance(surf_mesh, _basestring):
            if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                    surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                    surf_mesh.endswith('inflated')):
                coords, faces = nibabel.freesurfer.io.read_geometry(surf_mesh)
            elif surf_mesh.endswith('gii'):
                coords, faces = gifti.read(surf_mesh).darrays[0].data, \
                                gifti.read(surf_mesh).darrays[1].data
            else:
                raise ValueError('Format of mesh file not recognized.')
        # if a dictionary is given, check it contains entries for coords and faces
        elif isinstance(surf_mesh, dict):
            if ('faces' in surf_mesh and 'coords' in surf_mesh):
                coords, faces = surf_mesh['coords'], surf_mesh['faces']
            else:
                raise ValueError('If surf_mesh is given as a dictionary it must '
                                 'contain items with keys "coords" and "faces"')
        else:
            raise ValueError('surf_mesh must be a either filename or a dictionary '
                             'containing items with keys "coords" and "faces"')
        return coords, faces

    def get_cortical_indices(self, index):

        import nibabel

        c = sorted(nibabel.freesurfer.io.read_label(self.labels[index]))

        return c

    def crop_img(self, fig, margin=False):
        # takes fig, returns image
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import os
        from PIL import Image

        plt.tight_layout()
        fig.savefig('./tempimage', dpi = 300, transparent = True)
        plt.close(fig)
        image_data = Image.open('./tempimage.png')
        image_data_bw = image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

        os.remove('./tempimage.png')

        return image_data_new

    def plot_surf_stat_map(self, hemi, stat_map=None, bg_map=None,
                           view='lateral', threshold=None, cmap='coolwarm',
                           alpha='auto', vmax=None, symmetric_cbar="auto",
                           bg_on_stat=False, darkness=1, gii_darray=0,
                           output_file=None, **kwargs):

        """ Plotting of surfaces with optional background and stats map
                Parameters
                ----------
                surf_mesh: Surface object (to be defined)
                hemi: Hemisphere to display
                stat_map: Surface data (to be defined) to be displayed, optional
                bg_map: Surface data object (to be defined), optional,
                    background image to be plotted on the mesh underneath the
                    stat_map in greyscale, most likely a sulcal depth map for
                    realistic shading.
                view: {'lateral', 'medial', 'dorsal', 'ventral'}, view of the
                    surface that is rendered. Default is 'lateral'
                threshold : a number, None, or 'auto'
                    If None is given, the image is not thresholded.
                    If a number is given, it is used to threshold the image:
                    values below the threshold (in absolute value) are plotted
                    as transparent.
                cmap: colormap to use for plotting of the stat_map. Either a string
                    which is a name of a matplotlib colormap, or a matplotlib
                    colormap object.
                alpha: float, alpha level of the mesh (not the stat_map). If 'auto'
                    is chosen, alpha will default to .5 when no bg_map ist passed
                    and to 1 if a bg_map is passed.
                vmax: upper bound for plotting of stat_map values.
                symmetric_cbar: boolean or 'auto', optional, default 'auto'
                    Specifies whether the colorbar should range from -vmax to vmax
                    or from vmin to vmax. Setting to 'auto' will select the latter if
                    the range of the whole image is either positive or negative.
                    Note: The colormap will always be set to range from -vmax to vmax.
                bg_on_stat: boolean, if True, and a bg_map is specified, the
                    stat_map data is multiplied by the background image, so that
                    e.g. sulcal depth is visible beneath the stat_map. Beware
                    that this non-uniformly changes the stat_map values according
                    to e.g the sulcal depth.
                darkness: float, between 0 and 1, specifying the darkness of the
                    background image. 1 indicates that the original values of the
                    background are used. .5 indicates the background values are
                    reduced by half before being applied.
                gii_darray: integer, only applies when stat_map is given as a
                    gii_file, specifies the index of the gii array in which the data
                    for the stat_map ist stored.
                output_file: string, or None, optional
                    The name of an image file to export the plot to. Valid extensions
                    are .png, .pdf, .svg. If output_file is not None, the plot
                    is saved to a file, and the display is closed.
                kwargs: extra keyword arguments, optional
                    Extra keyword arguments passed to matplotlib.pyplot.imshow
            """


        from nilearn._utils.compat import _basestring
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        from mpl_toolkits.mplot3d import Axes3D

        # load mesh and derive axes limits
        #coords, faces = check_surf_mesh(surf_mesh)
        limits = [self.coords_[hemi].min(), self.coords_[hemi].max()]

        # set view
        if hemi == 'rh':
            if view == 'lateral':
                elev, azim = 0, 0
            elif view == 'medial':
                elev, azim = 0, 180
            elif view == 'dorsal':
                elev, azim = 90, 0
            elif view == 'ventral':
                elev, azim = 270, 0
            else:
                raise ValueError('view must be one of lateral, medial, '
                                 'dorsal or ventral')
        elif hemi == 'lh':
            if view == 'medial':
                elev, azim = 0, 0
            elif view == 'lateral':
                elev, azim = 0, 180
            elif view == 'dorsal':
                elev, azim = 90, 0
            elif view == 'ventral':
                elev, azim = 270, 0
            else:
                raise ValueError('view must be one of lateral, medial, '
                                 'dorsal or ventral')
        else:
            raise ValueError('hemi must be one of rh or lh')

        # set alpha if in auto mode
        if alpha == 'auto':
            if bg_map is None:
                alpha = .5
            else:
                alpha = 1

        # if cmap is given as string, translate to matplotlib cmap
        if isinstance(cmap, _basestring):
            cmap = plt.cm.get_cmap(cmap)

        # initiate figure and 3d axes
        fig = plt.figure(figsize = (20,14))
        ax = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        # plot mesh without data
        p3dcollec = ax.plot_trisurf(self.coords_[hemi][:, 0], self.coords_[hemi][:, 1], self.coords_[hemi][:, 2],
                                    triangles=self.faces_[hemi], linewidth=0.,
                                    antialiased=False,
                                    color='white')

        # If depth_map and/or stat_map are provided, map these onto the surface
        # set_facecolors function of Poly3DCollection is used as passing the
        # facecolors argument to plot_trisurf does not seem to work
        if bg_map is not None or stat_map is not None:

            face_colors = np.ones((self.faces_[hemi].shape[0], 4))
            face_colors[:, :3] = .5*face_colors[:, :3]

            if bg_map is not None:
                bg_data = self.bgs_[hemi]
                if bg_data.shape[0] != self.coords_[hemi].shape[0]:
                    raise ValueError('The bg_map does not have the same number '
                                     'of vertices as the mesh.')
                bg_faces = np.mean(bg_data[self.faces_[hemi]], axis=1)
                bg_faces = bg_faces - bg_faces.min()
                bg_faces = bg_faces / bg_faces.max()
                # control background darkness
                bg_faces *= darkness
                face_colors = plt.cm.gray_r(bg_faces)

            # modify alpha values of background
            face_colors[:, 3] = alpha*face_colors[:, 3]

            if stat_map is not None:
                #stat_map_data = self.check_surf_data(stat_map, gii_darray=gii_darray)
                stat_map_data = stat_map
                if stat_map_data.shape[0] != self.coords_[hemi].shape[0]:
                    raise ValueError('The stat_map does not have the same number '
                                     'of vertices as the mesh. For plotting of '
                                     'rois or labels use plot_roi_surf instead')
                stat_map_faces = np.mean(stat_map_data[self.faces_[hemi]], axis=1)

                # Call _get_plot_stat_map_params to derive symmetric vmin and vmax
                # And colorbar limits depending on symmetric_cbar settings
                cbar_vmin, cbar_vmax, vmin, vmax = \
                    self._get_plot_stat_map_params(stat_map_faces, vmax,
                                              symmetric_cbar, kwargs)

                #vmin = 0

                if threshold is not None:
                    kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                    stat_map_faces = stat_map_faces - vmin
                    stat_map_faces = stat_map_faces / (vmax-vmin)
                    if bg_on_stat:
                        face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                    else:
                        face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
                else:
                    stat_map_faces = stat_map_faces - vmin
                    stat_map_faces = stat_map_faces / (vmax-vmin)
                    if bg_on_stat:
                        face_colors = cmap(stat_map_faces) * face_colors
                    else:
                        face_colors = cmap(stat_map_faces)

            p3dcollec.set_facecolors(face_colors)

        # save figure if output file is given
        if output_file is not None:
            fig.savefig(output_file)
            plt.close(fig)
        else:
            return fig


    def add_plots(self, data, name = None, bg = True, view = 'all', hemi = 'both', cmap = 'jet'):

        # Set up initial variables

        if name is None:
            inds = [int(s.split('_')[-1]) for s in self.plots_.keys() if s.startswith('map')]
            if len(inds) > 0:
                name = 'map_%s' % str(np.max(inds)+1).zfill(3)
            else:
                name = 'map_001'

        if view == 'all':
            views = ['medial','lateral','dorsal','ventral']
        else:
            views = view

        # Indices for data to be plotted on respective hemis

        NV = self.coords_['lh'].shape[0]

        if hemi == 'both':
            hemis = ['lh','rh']
            self.hind_['lh'] = range(0,NV)
            self.hind_['rh'] = range(NV, NV*2)
        else:
            hemis = [hemi]
            self.hind_['lh'] = range(0,NV)
            self.hind_['rh'] = range(0,NV)

        if name not in self.plots_.keys():
            self.plots_[name] = {}


        # Do the actual plotting

        for v in views:
            if v not in self.plots_[name].keys():
                self.plots_[name][v] = {}

            for h in hemis:

                if data is not None:
                    d = data[self.hind_[h]]
                else:
                    d = data

                self.plots_[name][v][h] = self.plot_surf_stat_map(stat_map = d, bg_map = bg, hemi = h, view = v, cmap = cmap, bg_on_stat = True, symmetric_cbar = False, vmax = self.dmax)
                #plt.close(self.plots_[name][v][h])

    def remove_plot(self, name = None):

        if name is not None:
            try:
                del self.plots_[name]
            except KeyError:
                pass

    def save_plots(self, names = None, output_path = None):

        import os

        if output_path is not None and not os.path.isdir(output_path):
            os.mkdir(output_path)

        if names is not None:

            if names == 'all':
                names = self.plots_.keys()

            for n in names:
                try:
                    for v in self.plots_[n]:
                        for h in self.plots_[n][v].keys():
                            self.plots_[n][v][h].savefig(os.path.join(output_path, '%s_%s_%s.eps' % (n, v, h)))
                except KeyError:
                    pass
