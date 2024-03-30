# input one image and get depth information
import sys
import matplotlib.pyplot as plt
import hydra
# import open3d as o3d
from pathlib import Path
import math
import numpy as np

from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train import MLP
from digit_depth.train.prepost_mlp import *
from attrdict import AttrDict
from digit_depth.third_party import vis_utils
from digit_depth.handlers import find_recent_model, find_background_img
from scipy.spatial import ConvexHull

seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.resolve()


# edges = [[x1,y1,x2,y2],[...]]
# point = (x,y)

# above in a sense of the view in the output image
def point_above_line(x, y, line):
    rho, theta = line
    #print((rho + x * math.cos(theta) - y * math.sin(theta)) > 0)
    return (rho + x * math.cos(theta) - y * math.sin(theta)) > 0
def is_inside(line_tr, x, y):
    count = 0
    for line in line_tr:

        # Ray direction doesn't matter, choose y-axis
        if point_above_line(x, y, line) == point_above_line(x, 1000000, line):
            count += 1
    print(count)
    if count % 2 == 1:
        return False
    else:
        return True

def a_in_point(points, lines):
    for i in range(len(points[0])-1):
        print(i)
        print(points[0][i], points[1][i])
        if is_inside(lines, points[0][i], points[1][i]):
            return points[0][i], points[1][i]
    return None
# get the in_direction for each line if above(1) or below(-1)(in the sense of the view in the output image)

def get_in_direction(points,lines):
    x,y = a_in_point(points,lines)
    in_direction = []
    for line in lines:
        in_direction.append(point_above_line(x,y,line))


    print(in_direction)

    return in_direction


#@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def get_depth_distribution(cfg,img):
    # view_params = AttrDict({'fov': 60, 'front': [-0.1, 0.1, 0.1], 'lookat': [
    #     -0.001, -0.01, 0.01], 'up': [0.04, -0.05, 0.190], 'zoom': 2.5})
    # view_params = AttrDict({
    #             "fov": 60,
    #             "front": [-0.3, 0.0, 0.5],
    #             "lookat": [-0.001, 0.001,-0.001],
    #             "up": [0.0, 0.0, 0.50],
    #             "zoom": 0.5,
    #         })
    # vis3d = vis_utils.Visualizer3d(base_path=base_path, view_params=view_params)

    # projection params
    proj_mat = torch.tensor(cfg.sensor.P)
    model_path = find_recent_model(f"{base_path}/models")
    print(model_path)
    model = torch.load(model_path).to(device)
    model.eval()
    # base image depth map
    background_img_path = find_background_img(base_path)
    background_img = cv2.imread(background_img_path)
    background_img = preproc_mlp(background_img)
    background_img_proc = model(background_img).cpu().detach().numpy()
    background_img_proc, _ = post_proc_mlp(background_img_proc)
    # get gradx and grady
    gradx_base, grady_base = geom_utils._normal_to_grad_depth(img_normal=background_img_proc,
                                                              gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=None)

    # reconstruct depth
    img_depth_base = geom_utils._integrate_grad_depth(gradx_base, grady_base, boundary=None, bg_mask=None,
                                                      max_depth=0.0237)
    img_depth_base = img_depth_base.detach().cpu().numpy()  # final depth image for base image
    # setup digit sensor
    # digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    # digit_call = digit()
    # while True:

    # # read frame from images folder
    # frame = cv2.imread(f'{base_path}/images/mark_4.png')
    # check if frame is read
    if img is None:
        print('Error reading frame')
        sys.exit(1)
    # frame = digit_call.get_frame()
    img_np = preproc_mlp(img)
    img_np = model(img_np).detach().cpu().numpy()
    img_np, _ = post_proc_mlp(img_np)
    # get gradx and grady
    gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,
                                                            gel_height=cfg.sensor.gel_height, bg_mask=None)
    # reconstruct depth
    img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,
                                                 max_depth=cfg.max_depth)
    view_mat = torch.eye(4)  # torch.inverse(T_cam_offset)

    # the shape and meaning of each dimension of im_dpeth
    # print(img_depth.shape)
    numpy_array = img_depth.numpy()
    # print(numpy_array)

    # normalize the numpy array to 0-1
    # subtract the min value and divide by the max value
    numpy_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min())

    # show histogram of numpy array
    # Flatten the 2D array
    # flattened_array = numpy_array.flatten()
    # plt.hist(flattened_array, bins=100)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of 2D Numpy Array')
    # plt.show()

    # retain the 30 percent smallest value and set the remaining to 1
    numpy_array[numpy_array > np.percentile(numpy_array, 100)] = 1
    # turn the numpy array to a grey scale image
    numpy_array = 1 - numpy_array
    numpy_array = numpy_array * 255
    image = numpy_array.astype(np.uint8)



    # # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # # Find the contours using OpenCV
    # contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Draw the contours on the original image (for visualization)
    # image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    #
    # # Display the original image and the image with contours
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Image with Contours", image_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Apply Canny edge detection
    # Apply median blur to reduce noise
    blurred_image = cv2.medianBlur(image, 5)  # Adjust kernel size as needed

    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    # Apply erosion to refine edges and remove noise
    eroded_image = cv2.erode(blurred_image, kernel, iterations=1)
    # show
    cv2.imshow('Eroded Image', eroded_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # find contour
    ret, thresh = cv2.threshold(eroded_image, 160, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # show the contours (for visualization)
    cv2.drawContours(eroded_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', eroded_image)
    # cv2.waitKey(0)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    max_contour = contours[max_index]

    # draw approx
    epsilon = 0.1 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # draw the polygon in red

    # Create blank color image
    color_img = np.ones_like(eroded_image) * 255

    # # Draw contour
    # cv2.drawContours(color_img, [approx], -1, (0, 255, 0), 2)
    #
    # # Show result
    cv2.imshow('Approx polygon in red', color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # identify the line := has the largest distance among all lines to their furthest point in object(in erode_image)
    out_nums = []

    points_in_obj = eroded_image.nonzero()
    print(points_in_obj)
    vertexes = approx[:, 0, :]
    # print(len(lines))
    # append the first point to the end of the list
    vertexes = np.append(vertexes, [vertexes[0]], axis=0)
    # # append the second point to the end of the list
    # vertexes = np.append(vertexes, [vertexes[1]], axis=0)
    #vertexes = np.append(vertexes, [vertexes[2]], axis=0)
    ## so that line_tr would contain 2 more lines, and we access the 1~n-2 index

    # calculate the distances between vertexes
    dists = []
    for i in range(len(vertexes) - 1):
        x1, y1 = vertexes[i]
        x2, y2 = vertexes[i + 1]
        dists.append(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    # print(len(lines))
    lines_tr = []
    # print the points

    # print vertex
    # for i in range(len(lines)-1):
    #     print(lines[i])

    # # draw point(57,185) on a blank image in the size of the original image
    # blank_image = np.zeros((eroded_image.shape[0], eroded_image.shape[1], 3), np.uint8)
    # cv2.circle(blank_image, (57, 185), 1, (255, 255, 255), 1)
    # cv2.imshow('point', blank_image)
    # cv2.waitKey(0)
    for i in range(len(vertexes) - 1):
        x1, y1 = vertexes[i]
        x2, y2 = vertexes[i + 1]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        theta = math.atan(m)
        rho = b / math.cos(theta)
        lines_tr.append((rho, theta))
        print(rho, theta)
    print(len(lines_tr))
    in_direction = get_in_direction(points_in_obj, lines_tr)

    # for j in range(1, len(lines_tr) - 2):
    #     rho, theta = lines_tr[j]
    #     print(rho, theta)
    #     num = 0
    #     for i in range(len(points_in_obj[0] - 1)):
    #         x, y = points_in_obj[0][i], points_in_obj[1][i]
    #         if is_inside(lines_tr, x, y):
    #             continue
    #         if point_above_line(x, y, lines_tr[j]):
    #             num += 1
    #
    #     out_nums.append(num)
    # print(out_nums)
    #
    # max_line = lines_tr[out_nums.index(max(out_nums))]
    #x, y = max_point_pose[max_dists.index(max(max_dists))]

    # select the longest line
    max_line = lines_tr[dists.index(max(dists))]
    rho, theta = max_line
    up = in_direction[dists.index(max(dists))]

    print(theta, up)
    return theta, up
    # cv2.polylines(eroded_image,[approx],True,(0,0,255),2)
    # cv2.imshow('Largest Polygon', eroded_image)
    # cv2.waitKey(0)

    # # draw convex hull
    # coords = np.column_stack(np.where(eroded_image > 0))
    # hull = ConvexHull(coords)
    # cv2.polylines(eroded_image,[hull.simplices.copy()],True,(255))    #
    # cv2.imshow('Convex Hull', eroded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # edges = cv2.Canny(eroded_image, 20, 150, apertureSize=3)
    #
    # # Display the result
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # Apply Hough Line Transform
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=10)
    #
    # # show all the lines found by hough
    # # Find the most significant line based on accumulator value
    # if lines is not None:
    #     most_significant_line = max(lines, key=lambda x: x[0][0])
    #     rho, theta = most_significant_line[0]
    #     print("Most significant line parameters: rho =", rho, "theta =", theta)
    # else:
    #     print("No lines detected.")
    #
    # # Draw the most significant line on the original image
    # line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # x1 = int(x0 + 1000 * (-b))
    # y1 = int(y0 + 1000 * (a))
    # x2 = int(x0 - 1000 * (-b))
    # y2 = int(y0 - 1000 * (a))
    # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # # Display the result
    # cv2.imshow('Most Significant Line', line_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=view_mat, params=cfg.sensor)
    # points3d = geom_utils.remove_background_pts(points3d, bg_mask=None)
    # cloud = o3d.geometry.PointCloud()
    # clouds = geom_utils.init_points_to_clouds(clouds=[copy.deepcopy(cloud)], points3d=[points3d])
    # vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds)


if __name__ == "__main__":
    get_depth_distribution()
