import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skopi as sk
from extra_geom import AGIPD_1MGeometry


def get_skopi_sensor(
                       input_detector: AGIPD_1MGeometry,
                       distance: float,
                       beam: sk.Beam,
                       single_pixel_height: float = 0.0002,
                       single_pixel_width: float = 0.0002,
                       simulate: bool = False,
                       particle: sk.Particle = None,
                       ) -> sk.UserDefinedDetector:
    """
    Generates and returns a skopi.UserDefinedDetector object from a
     extra_geom Geometry object. Requires the sensor pixel size,
     defaults to 0.2mm (AGIPD-1M).

    Parameters:
    input_detector: The input object.
    beam: The skopi.Beam object to use with the detector. Optional.
    pixel_height: The height of a single pixel, in meters.
    pixel_width: The width of a single pixel, in meters.
    simulate: Toggles a short simulation which shows the sensor visually.
     Defaults to False, requires a GPU with CUDA.
    particle: Only needed if simulate is True. The skopi.Particle object
     to use in the simulation. A basic particle can be found at
     https://github.com/chuckie82/skopi/blob/main/examples/input/pdb/2cex.pdb.
    """
    pixel_pos = input_detector.get_pixel_positions()
    # pixel_matrix = input_detector.to_distortion_array()
    # pixel_matrix_pixels = np.around(pixel_matrix / single_pixel_width, 0)
    dummy = np.ones(input_detector.expected_data_shape)
    data_dummy,center = input_detector.position_modules_fast(dummy)
    plt.figure()
    plt.imshow(data_dummy[::-1,::-1])
    plt.savefig("dummp.png",dpi=300)


    p_center_x = -pixel_pos[:, :, :, 0]  # only gets x coords
    p_center_y = -pixel_pos[:, :, :, 1]  # only gets y coords

    # p_map = pixel_matrix_pixels.mean(axis=2)[:, :, 1:] \
        # .reshape(pixel_pos.shape[0], pixel_pos.shape[1], pixel_pos.shape[2], 2)
    p_map_shape = input_detector.expected_data_shape+(2,)
    p_map = np.empty(p_map_shape)
    # pixel_x = np.round(p_center_x/single_pixel_width+center[1],2)
    pixel_x = np.floor(p_center_x/single_pixel_width+center[1])
    pixel_y = np.floor(p_center_y/single_pixel_height+center[0])
    # pixel_y = np.round(p_center_y/single_pixel_height+center[0],2)
    p_map[...,1] = pixel_x - pixel_x.min()
    p_map[...,0] = pixel_y - pixel_y.min()

    pixel_height_array = single_pixel_height * np.ones(p_center_y.shape)
    pixel_width_array = single_pixel_width * np.ones(p_center_x.shape)

    detector_geometry = {
        'panel number': pixel_pos.shape[0],
        'panel pixel num x': pixel_pos.shape[1],
        'panel pixel num y': pixel_pos.shape[2],
        'detector distance': distance,  # distance between detector and sample in m
        'pixel width': pixel_width_array,  # width of each pixel as array
        'pixel height': pixel_height_array,  # height of each pixel as array
        'pixel center x': p_center_x,  # x-coordinate of each pixel center
        'pixel center y': p_center_y,  # y-coordinate of each pixel center
        'pixel map': p_map,  # map to assemble detector
    }
    if not simulate:
        return sk.UserDefinedDetector(geom=detector_geometry, beam=beam)
    else:
        my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis')) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        detector = sk.UserDefinedDetector(geom=detector_geometry, beam=beam)
        experiment = sk.SPIExperiment(detector, beam, particle)
        dataset = np.zeros((1,) + detector.shape, np.float32)
        # beam = sk.Beam(photon_energy=4600, fluence=1e12, focus_radius=1e-7)
        photons = np.zeros((1,) + detector.shape, np.int32)
        dataset[0] = experiment.generate_image_stack(return_intensities=True)
        N = dataset.shape[0]
        print(N)
        plt.figure(figsize=(20, 15/N+1))
        for i in range(N):
            plt.subplot(1, N, i+1)
            img = experiment.det.assemble_image_stack(dataset[i])
            plt.imshow(img, norm=matplotlib.colors.LogNorm(),cmap=my_cmap)
            print(data_dummy.shape)
            print(img.shape)
            # assert img.shape == data_dummy.shape
        # plt.show()
        plt.colorbar()
        plt.savefig("./pattern_2.png",dpi=300)
        # import pdb; pdb.set_trace()
        return detector


# sample_agipd = AGIPD_1MGeometry.from_quad_positions(quad_pos=[(-525, 625),
                                                              # (-550, -10),
                                                              # (520, -160),
                                                              # (542.5, 475)])
sample_agipd = AGIPD_1MGeometry.from_crystfel_geom("./agipd_2934_v6_121.geom")
sample_agipd.inspect()
plt.savefig("./inspect.png",dpi=300)
sample_beam = sk.Beam(photon_energy=9300, fluence=1e12, focus_radius=1e-7)
sample_particle = sk.Particle()
sample_particle.read_pdb('2CEX.pdb', ff='WK')
get_skopi_sensor(input_detector=sample_agipd,
                 distance = 0.7,
                 single_pixel_width=0.0002,
                 single_pixel_height=0.0002,
                 simulate=True,
                 beam=sample_beam,
                 particle=sample_particle)
