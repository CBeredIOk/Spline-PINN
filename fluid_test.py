import cv2
import copy
import torch
import time, os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy2vtk import imageToVTK
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Logger import Logger
from fluid_setups import Dataset
from operators import vector2HSV
from get_param import params, toCuda, toCpu, get_hyperparam_fluid
from spline_models import superres_2d_velocity, superres_2d_pressure, get_Net, interpolate_states


def load_images_from_folder(folder="imgs", threshold=100):
    """
    Downloads images from a folder and returns:
      - image_names: list[str] with images names
      - images: dict, where the key is the name (without extension), the value is the tensor mask
    """
    images = {}
    image_names = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            name = os.path.splitext(filename)[0]
            image_names.append(name)
            path = os.path.join(folder, filename)
            img_arr = np.asarray(Image.open(path))
            tensor = (torch.mean(torch.tensor(img_arr).float(), dim=2) < threshold).float()
            images[name] = tensor
    return image_names, images

# Getting a dictionary of images and list of images names
image_names, images = load_images_from_folder("imgs")

if params.type == "image" and params.image is not None:
    image_names = [image for image in image_names if image == params.image]
    images = {key: value for key, value in images.items() if key == params.image}


# python fluid_test.py --net=Fluid_model --hidden_size=50 --mu=0.1 --rho=10

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allows OpenMP duplication (temporary)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

save_movie=False # True
movie_FPS = 30.0 # 8.0 # 15.0 # ... choose FPS as provided in visualization
# params.width = 230 if params.width is None else params.width
# params.height = 49 if params.height is None else params.height
params.width = 230 if params.width is None else params.width
params.height = 49 if params.height is None else params.height
resolution_factor = params.resolution_factor
orders_v = [params.orders_v,params.orders_v]
orders_p = [params.orders_p,params.orders_p]
v_size,p_size = np.prod([i+1 for i in orders_v]),np.prod([i+1 for i in orders_p])
mouse_paint,mouse_erase = False,False

# dataset types to randomly choose from
types = [params.type] # interactive paint, DFG benchmark and Magnus effect environment. Further types to choose from: "box","image","ecmo","poiseuille"

# initialize dataset
dataset = Dataset(
    params.width, params.height, hidden_size=v_size+p_size,
    interactive=True, batch_size=1, n_samples=1, dataset_size=1,
    average_sequence_length=params.average_sequence_length, types=types, dt=params.dt,
    images=images, image_names=image_names, resolution_factor=resolution_factor
)

# initialize windows / movies / mouse handler
# cv2.namedWindow('a_z',cv2.WINDOW_NORMAL)
cv2.namedWindow('Velocity magnitude', cv2.WINDOW_NORMAL)
cv2.namedWindow('Velocity direction', cv2.WINDOW_NORMAL)
cv2.namedWindow('Velocity direction legend', cv2.WINDOW_NORMAL)
cv2.namedWindow('Pressure', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Velocity magnitude', 240, 680)
cv2.resizeWindow('Velocity direction', 180, 680)
cv2.resizeWindow('Velocity direction legend', 180, 180)
cv2.resizeWindow('Pressure', 240, 680)

def save_results(tensor_name, tensor):
    tensor_np = tensor.cpu().detach().numpy()
    with open(f"tensor_{tensor_name}.txt", "w") as f:
        for i, slice_ in enumerate(tensor_np):  # 1 level
            for j, matrix in enumerate(slice_):  # 2 level
                f.write(f"Slice {i}, Matrix {j}:\n")
                np.savetxt(f, matrix, fmt="%.6f")
                f.write("\n")

if save_movie:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    movie_p = cv2.VideoWriter(f'plots/p_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))
    movie_v = cv2.VideoWriter(f'plots/v_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))
    movie_a = cv2.VideoWriter(f'plots/a_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))

def mousePosition(event, x, y, flags, param):
    global dataset
    if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1 or mouse_erase or mouse_paint:
        if mouse_erase:
            dataset.mouse_erase = True
        if mouse_paint:
            dataset.mouse_paint = True
        dataset.mousex = y/resolution_factor
        dataset.mousey = x/resolution_factor

# cv2.setMouseCallback("a_z",mousePosition)
cv2.setMouseCallback("Velocity magnitude", mousePosition)
cv2.setMouseCallback("Velocity direction", mousePosition)
cv2.setMouseCallback("Pressure", mousePosition)

vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)])
image = vector2HSV(vector)
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
cv2.imshow('Velocity direction legend', image)

# load fluid model
model = toCuda(get_Net(params))
logger = Logger(get_hyperparam_fluid(params),use_csv=False,use_tensorboard=False)
date_time,index = logger.load_state(model,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, index: {index}")
model.eval()

FPS = 0
last_FPS = 0
last_time = time.time()

# compute pressure and viscous forces
def forces(grad_v, p, x, n):
    """
    :grad_v: gradient of velocity field
    :p: pressure field
    :x: location of measurement (will be rounded to integers) [n_samples x 2]
    :n: surface normal [n_samples x 2]
    :return:
        :pressure_force: pressure force [n_samples x 2]
        :viscous_force: viscous force [n_samples x 2]
    """
    pos = x.long()
    grads_v = grad_v[0,:,pos[:,0],pos[:,1]].permute(1,0)
    grads_v = torch.cat([grads_v[:,0:2].unsqueeze(1),grads_v[:,2:4].unsqueeze(1)],dim=1)
    dv_dn = torch.matmul(grads_v,n.unsqueeze(2))
    ps = p[0,0,pos[:,0],pos[:,1]]
    ps = ps-torch.mean(ps)
    pressure_force = ps.unsqueeze(1)*n[:,:]
    viscous_force = params.mu*dv_dn[:,:,0]
    return -pressure_force.detach(),viscous_force.detach()


def make_colorbar(
    min_val, max_val,
    height=256,
    width=60,
    cmap=cv2.COLORMAP_JET,
    num_ticks=5,
    right_margin=140,
    top_offset=20,
    bottom_offset=20
):
    """
    Creates a scale image (colorbar) with height (for the gradient itself),
    plus top/bottom margins. Width = width + right_margin.
    The top_offset and bottom_offset can be adjusted so that it doesn't crop the lettering.
    """
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    colorbar = np.repeat(gradient, width, axis=1)
    colorbar = cv2.applyColorMap(colorbar, cmap)  # (height, width, 3)

    total_height = height + top_offset + bottom_offset
    total_width = width + right_margin

    bar_full = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    bar_full[top_offset : top_offset + height, 0 : width] = colorbar

    tick_values = np.linspace(min_val, max_val, num_ticks)
    for val in tick_values:
        fraction = (max_val - val) / (max_val - min_val)

        x_pos = width + 5

        y_pos = top_offset + int(fraction * (height - 1))

        cv2.putText(
            bar_full,
            f"{val:.2f}",
            (x_pos, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # scale of value on colorbar
            (255, 255, 255),
            2,  # thickness of value on colorbar
            cv2.LINE_AA
        )
    return bar_full


# simulation loop
exit_loop = False
while not exit_loop:

    # reset environment (choose new random environment from types-list and reset velocity / pressure field to 0)
    dataset.reset_env(0)

    # buffers for drag / lift coefficients
    cd_cl_t = torch.zeros(200,2)

    dataset.mousev = 1#1.5

    for i in range(params.average_sequence_length):
        #print(f"{epoch} / {i}")

        # obtain boundary conditions / mask as well as spline coefficients of previous timestep from dataset
        v_cond,v_mask,old_hidden_state,_,_,_ = toCuda(dataset.ask())

        # apply fluid model to obtain spline coefficients of next timestep
        new_hidden_state = model(old_hidden_state,v_cond,v_mask)

        # feed new spline coefficients back to the dataset
        dataset.tell(toCpu(new_hidden_state))


        # compute drag and lift coefficients
        if dataset.env_info[0]["type"] == "magnus" or dataset.env_info[0]["type"] == "DFG_benchmark":
            radius = dataset.env_info[0]["r"]
            x = torch.FloatTensor([[(dataset.env_info[0]["x"]+radius*np.cos(phi))*resolution_factor,(dataset.env_info[0]["y"]+radius*np.sin(phi))*resolution_factor] for phi in np.arange(0,2*np.pi,0.05)])
            n = torch.FloatTensor([[np.cos(phi),np.sin(phi)] for phi in np.arange(0,2*np.pi,0.05)])
            a_z,v,grad_v,laplace_v = superres_2d_velocity(new_hidden_state[0:1,:v_size],orders_v,resolution_factor)
            p,grad_p = superres_2d_pressure(new_hidden_state[0:1,v_size:],orders_p,resolution_factor)
            pressure_force,viscous_force = forces(grad_v.cpu(),p.cpu(),x,n)
            pressure_force = torch.mean(pressure_force,dim=0)*2*3.14*radius
            viscous_force = torch.mean(viscous_force,dim=0)*2*3.14*radius
            total_force = pressure_force+viscous_force
            cd_cl = total_force*2/(2*radius)/((dataset.mousev*2/3)**2)/params.rho
            cd_cl_t[0:-1,:] = cd_cl_t[1:,:].clone()
            cd_cl_t[-1] = cd_cl
            print(f"C_D / C_L = {cd_cl}")

        # visualize fields
        if i % 1 == 0:

            print(f"env_info: {dataset.env_info[0]}")

            # obtain interpolated field values for a_z,v,grad_v,laplace_v from spline coefficients of velocity field
            a_z, v, grad_v, laplace_v = superres_2d_velocity(new_hidden_state[0:1,:v_size],orders_v,resolution_factor)

            image = a_z[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach().clone()
            image = image - torch.min(image)
            image /= torch.max(image)
            image = toCpu(image).unsqueeze(2).repeat(1, 1, 3).numpy()

            # if save_movie:
            #     movie_a.write((255*image).astype(np.uint8))
            # cv2.imshow('a_z',image)

            mask_np = toCpu(dataset.v_mask_full_res[0, 0,
                            resolution_factor:-resolution_factor,
                            resolution_factor:-resolution_factor]).numpy()
            mask_bool = (mask_np == 1)

            vector = v[0, :, resolution_factor:-resolution_factor, resolution_factor:-resolution_factor].cpu().detach()

            hsv_image = vector2HSV(vector)
            direction_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            direction_image[mask_bool] = [0, 0, 0]

            if save_movie:
                movie_v.write((255 * direction_image).astype(np.uint8))
            cv2.imshow('Velocity direction', direction_image)

            v_magnitude = torch.sqrt(vector[0] ** 2 + vector[1] ** 2)
            v_magnitude = v_magnitude - torch.min(v_magnitude)
            v_magnitude = v_magnitude / torch.max(v_magnitude)

            gray_image = (255 * v_magnitude).type(torch.uint8).numpy()

            colored_magnitude = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

            colored_magnitude[mask_bool] = [0, 0, 0]

            real_min_val = float(torch.min(vector[0].hypot(vector[1])))
            real_max_val = float(torch.max(vector[0].hypot(vector[1])))

            height_img = colored_magnitude.shape[0]  # 1824

            bar_height = height_img // 2  # 912
            cb_magnitude = make_colorbar(
                real_min_val,
                real_max_val,
                height=bar_height,
                width=40,
                num_ticks=5
            )
            bar_canvas = np.zeros((height_img, cb_magnitude.shape[1], 3), dtype=np.uint8)
            offset = (height_img - cb_magnitude.shape[0]) // 2
            bar_canvas[offset:offset + cb_magnitude.shape[0], 0:cb_magnitude.shape[1]] = cb_magnitude

            colored_magnitude_with_bar = cv2.hconcat([colored_magnitude, bar_canvas])

            if save_movie:
                movie_v.write(colored_magnitude_with_bar)
            cv2.imshow('Velocity magnitude', colored_magnitude_with_bar)

            # obtain interpolated field values for p,grad_p from spline coefficients of pressure field
            p, grad_p = superres_2d_pressure(new_hidden_state[0:1, v_size:], orders_p, resolution_factor)
            p_for_visualization = p[0, 0, resolution_factor:-resolution_factor,
                                  resolution_factor:-resolution_factor].cpu().detach()
            mask = dataset.v_mask_full_res[0, 0, resolution_factor:-resolution_factor,
                   resolution_factor:-resolution_factor].cpu()

            p_numpy = p_for_visualization.numpy()
            mask_np = mask.numpy().astype(bool)
            valid_values = p_numpy[~mask_np]

            real_min_p = float(valid_values.min())
            real_max_p = float(valid_values.max())

            p_norm = (p_for_visualization - real_min_p) / (real_max_p - real_min_p)
            p_norm[mask == 1] = 0

            gray_image = (255 * toCpu(p_norm)).type(torch.uint8).numpy()
            colored_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

            height_img = colored_image.shape[0]
            bar_height = height_img // 2

            cb_pressure = make_colorbar(
                min_val=real_min_p,
                max_val=real_max_p,
                height=bar_height,
                width=40,
                cmap=cv2.COLORMAP_JET,
                num_ticks=5
            )

            bar_canvas = np.zeros((height_img, cb_pressure.shape[1], 3), dtype=np.uint8)
            offset = (height_img - cb_pressure.shape[0]) // 2
            bar_canvas[offset:offset + cb_pressure.shape[0], 0:cb_pressure.shape[1]] = cb_pressure

            colored_image[mask_np] = [0, 0, 0]

            pressure_with_bar = cv2.hconcat([colored_image, bar_canvas])

            if save_movie:
                movie_p.write(pressure_with_bar)
            cv2.imshow('Pressure', pressure_with_bar)

            key = cv2.waitKey(1)

            if key==ord('x'): # increase flow velocity
                dataset.mousev+=0.1
            elif key==ord('y'): # decrease flow velocity
                dataset.mousev-=0.1

            elif key==ord('1'): # different flow velocities...
                dataset.mousev=0.3
            elif key==ord('2'):
                dataset.mousev=0.5
            elif key==ord('3'):
                dataset.mousev=1
            elif key==ord('4'):
                dataset.mousev=1.5

            if key==ord('s'): # increase spin of cylinder
                dataset.mousew+=0.1
            elif key==ord('a'): # decrease spin of cylinder
                dataset.mousew-=0.1

            elif key==ord('r'): # reset position of cylinder
                dataset.mousex=25
                dataset.mousey=24

            if key==ord('w'): # 'write' in paint environment
                mouse_paint=True
            else:
                dataset.mouse_paint=False
                mouse_paint=False
            if key==ord('e'): # 'erase' in paint environment
                mouse_erase=True
            else:
                dataset.mouse_erase=False
                mouse_erase=False

            if key==ord('p'):

                os.makedirs("plots", exist_ok=True)

                v_np = toCpu(v).numpy()
                v_x = v_np[0, 0, resolution_factor:-resolution_factor, resolution_factor:-resolution_factor]
                v_y = v_np[0, 1, resolution_factor:-resolution_factor, resolution_factor:-resolution_factor]

                figsize = (12, 4)

                mask_np = toCpu(dataset.v_mask_full_res[0, 0,
                                resolution_factor:-resolution_factor,
                                resolution_factor:-resolution_factor]).numpy()

                v_x_masked = np.ma.masked_where(mask_np == 1, v_x)
                v_y_masked = np.ma.masked_where(mask_np == 1, v_y)

                jet_cmap = copy.copy(plt.cm.jet)
                jet_cmap.set_bad('k', 1.0)

                v_x_masked = np.rot90(v_x_masked, k=1)
                v_y_masked = np.rot90(v_y_masked, k=1)

                # Plotting for the X component of velocity
                fig1, ax1 = plt.subplots(figsize=figsize)
                im1 = ax1.imshow(v_x_masked, cmap=jet_cmap, origin='lower',
                                 vmin=np.min(v_x_masked), vmax=np.max(v_x_masked))
                ax1.set_title("X velocity")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.invert_yaxis()
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label("Value v_x")
                plt.tight_layout()
                plt.savefig(f"plots/velocity_x_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", dpi=300)

                # Plotting for the Y component of velocity
                fig2, ax2 = plt.subplots(figsize=figsize)
                im2 = ax2.imshow(v_y_masked, cmap=jet_cmap, origin='lower',
                                 vmin=np.min(v_y_masked), vmax=np.max(v_y_masked))
                ax2.set_title("Y velocity")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.invert_yaxis()
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label("Value v_y")
                plt.tight_layout()
                plt.savefig(f"plots/velocity_y_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", dpi=300)

                os.makedirs("plots",exist_ok=True)
                name = dataset.env_info[0]["type"]
                if name=="image":
                    name = name+"_"+dataset.env_info[0]["image"]

                save_results("v", v)
                save_results("p", p)
                save_results("a_z", a_z)

                # Create pressure plot with streamlines
                flow = v[0,:,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach().clone()
                image = vector2HSV(flow)
                flow = toCpu(flow).numpy()
                fig3 = plt.figure(3, figsize=(12, 4))
                fig3.tight_layout()
                ax3 = fig3.add_subplot()
                ax3.set_title("Pressure with streamlines")

                Y,X = np.mgrid[0:flow.shape[1], 0:flow.shape[2]]
                linewidth = image[:,:,2] / np.max(image[:,:,2])
                ax3.streamplot(Y.transpose(),X.transpose(),  flow[0].transpose()[::-1], -flow[1].transpose()[::-1], color='k', density=3,linewidth=2*linewidth.transpose()[::-1])

                # Standard palette
                palette = plt.cm.gnuplot2
                palette.set_bad('k',1.0)

                # ANSYS palette
                jet_cmap = copy.copy(plt.cm.jet)
                jet_cmap.set_bad('k', 1.0)

                p = p[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach()
                cond_mask = dataset.v_mask_full_res[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor]
                p_m = np.ma.masked_where(toCpu(cond_mask).numpy()==1, toCpu(p).numpy())
                # plt.imshow(pm.transpose()[::-1],cmap=palette)
                plt.imshow(p_m.transpose()[::-1], cmap=jet_cmap)
                plt.axis('off')
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes("right",size="5%",pad=0.05)
                plt.colorbar(cax=cax)
                plt.savefig(f"plots/flow_and_pressure_field_{name}_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", bbox_inches='tight',dpi=300)

                v_norm = np.linalg.norm(flow, axis=0)

                cond_mask_np = toCpu(dataset.v_mask_full_res[0, 0,
                                     resolution_factor:-resolution_factor,
                                     resolution_factor:-resolution_factor]).numpy()

                v_norm_m = np.ma.masked_where(cond_mask_np == 1, v_norm)

                # Create velocity magnitude plot
                fig4 = plt.figure(4, figsize=(12, 4))
                fig4.tight_layout()
                ax4 = fig4.add_subplot()
                ax4.set_title("Velocity magnitude")

                # plt.imshow(np.linalg.norm(flow,axis=0).transpose()[::-1])
                # plt.imshow(np.linalg.norm(flow, axis=0).transpose()[::-1], cmap='jet')
                plt.imshow(v_norm_m.transpose()[::-1], cmap=jet_cmap)
                plt.axis('off')
                divider = make_axes_locatable(ax4)
                cax = divider.append_axes("right",size="5%",pad=0.05)
                plt.colorbar(cax=cax)

                """
                # save results in vtk files
                os.makedirs(f"vtk/{name}/{get_hyperparam_fluid(params)}",exist_ok=True)
                p -= torch.mean(p)
                imageToVTK(f"vtk/{name}/{get_hyperparam_fluid(params)}/pressure",pointData={"pressure":(p*(1-cond_mask)).unsqueeze(2).numpy()})
                v_new = (v[0,:,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu()*(1-cond_mask)).unsqueeze(3).detach()
                imageToVTK(f"vtk/{name}/{get_hyperparam_fluid(params)}/velocity",cellData={"velocity":(v_new[0].numpy(),v_new[1].numpy(),0*v_new[1].numpy())})
                """

                if dataset.env_info[0]["type"] == "magnus" or dataset.env_info[0]["type"] == "DFG_benchmark":

                    # print min / average / max for c_d / c_l
                    print(f"C_D: (min: {torch.min(cd_cl_t[:,0])} / avrg: {torch.mean(cd_cl_t[:,0])} / max: {torch.max(cd_cl_t[:,0])}) ; C_L: (min: {torch.min(cd_cl_t[:,1])} / avrg: {torch.mean(cd_cl_t[:,1])} / max: {torch.max(cd_cl_t[:,1])})")

                    # compute c_d / c_l again
                    p,_ = superres_2d_pressure(new_hidden_state[0:1,v_size:],orders_p,resolution_factor)
                    pressure_force,viscous_force = forces(grad_v.cpu(),p.cpu(),x,n)
                    total_force = torch.mean(pressure_force+viscous_force,dim=0)*2*3.14*radius
                    cd_cl = total_force*2/(2*radius)/((dataset.mousev*2/3)**2)

                    # visualize pressure forces:
                    palette = plt.cm.gnuplot2
                    palette.set_bad('k',1.0)

                    jet_cmap = copy.copy(plt.cm.jet)
                    jet_cmap.set_bad('k', 1.0)

                    p = p[0,0].cpu().detach()
                    cond_mask = dataset.v_mask_full_res[0,0]

                    flow = v[0,:].cpu().detach().clone()
                    flow[:,:resolution_factor] = 0
                    flow[:,:,:resolution_factor] = 0
                    flow[:,-resolution_factor:] = 0
                    flow[:,:,-resolution_factor:] = 0
                    flow = toCpu(flow).numpy()

                    v_norm = np.linalg.norm(flow, axis=0)
                    v_m = np.ma.masked_where(cond_mask == 1, v_norm)
                    p_m = np.ma.masked_where(toCpu(cond_mask).numpy() == 1, toCpu(p).numpy())

                    fig5 = plt.figure(5, figsize=figsize)
                    plt.clf()
                    ax5_1 = fig5.add_subplot(1,2,2)
                    # plt.imshow(p_m.transpose(), cmap=palette)
                    plt.imshow(p_m.transpose(), cmap=jet_cmap)
                    pressure_force/= torch.max(pressure_force)
                    pressure_force*=resolution_factor
                    for j in range(0,x.shape[0],2):
                        plt.arrow(x[j,0],x[j,1],pressure_force[j,0],pressure_force[j,1],head_width=0.4*resolution_factor,head_length=0.4*resolution_factor,zorder=10,fc='w',ec='w')
                    plt.plot(torch.cat([x[:,0],x[0:1,0]]),torch.cat([x[:,1],x[0:1,1]]),color='w')
                    plt.xlim((torch.min(x[:,0])-30),(torch.max(x[:,0])+30))
                    plt.ylim((torch.min(x[:,1])-30),(torch.max(x[:,1])+30))
                    plt.title("Pressure forces")
                    plt.xlabel("x axis")
                    plt.ylabel("y axis")
                    plt.axis('off')
                    divider = make_axes_locatable(ax5_1)
                    cax = divider.append_axes("right",size="5%",pad=0.05)
                    plt.colorbar(cax=cax)

                    # visualize viscous forces:
                    ax5_2 = fig5.add_subplot(1,2,1)
                    # plt.imshow(np.linalg.norm(flow,axis=0).transpose())
                    # plt.imshow(np.linalg.norm(flow, axis=0).transpose(), cmap=jet_cmap)
                    plt.imshow(v_m.transpose(), cmap=jet_cmap)
                    viscous_force/= torch.max(viscous_force)
                    viscous_force*=resolution_factor
                    for j in range(0,x.shape[0],2):
                        plt.arrow(x[j,0],x[j,1],viscous_force[j,0],viscous_force[j,1],head_width=0.4*resolution_factor,head_length=0.4*resolution_factor,zorder=10,fc='w',ec='w')
                    plt.plot(torch.cat([x[:,0],x[0:1,0]]),torch.cat([x[:,1],x[0:1,1]]),color='w')
                    plt.xlim((torch.min(x[:,0])-30),(torch.max(x[:,0])+30))
                    plt.ylim((torch.min(x[:,1])-30),(torch.max(x[:,1])+30))
                    plt.title("Viscous forces")
                    plt.xlabel("x axis")
                    plt.ylabel("y axis")
                    plt.axis('off')
                    divider = make_axes_locatable(ax5_2)
                    cax = divider.append_axes("right",size="5%",pad=0.05)
                    plt.colorbar(cax=cax)
                    plt.savefig(f"plots/pressure_viscous_forces_{name}_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", bbox_inches='tight',dpi=300)

                    # plot drag and lift coefficients over time
                    plt.figure(6)
                    plt.plot(cd_cl_t[:,0])
                    plt.plot(cd_cl_t[:,1])
                    plt.title("$C_D$ / $C_L$ over time")
                    plt.legend(["$C_D$","$C_L$"])
                    plt.xlabel("time")
                    plt.ylabel("$C_D$ / $C_L$")

                    # Struhal number (St) calculation based on the lift coefficient time series
                    # It is assumed that cd_cl_t is a tensor of size [N, 2],
                    # where the second component (index 1) corresponds to the lift coefficient (C_L)
                    cl_series = cd_cl_t[:, 1].detach().cpu().numpy()  # extract time series C_L
                    print(f"cl_series = {cl_series}")

                    cl_series = cl_series - np.mean(cl_series)  # remove the constant component

                    # Determine the sampling rate
                    # If dt is the simulation time step (parameter params.dt), then:
                    fs = 1.0 / params.dt  # sampling frequency in Hz

                    # Apply FFT
                    n = len(cl_series)
                    fft_vals = np.fft.fft(cl_series)
                    fft_freqs = np.fft.fftfreq(n, d=params.dt)

                    # Keep only positive frequencies
                    pos_mask = fft_freqs > 0
                    fft_freqs = fft_freqs[pos_mask]
                    fft_magnitudes = np.abs(fft_vals[pos_mask])

                    # Determine the dominant frequency (vortex breakaway frequency)
                    dominant_freq = fft_freqs[np.argmax(fft_magnitudes)]

                    # Determine the characteristic dimension: for a cylinder L = 2r
                    r = dataset.env_info[0].get("r", 1) # if the parameter r is missing, default to 1
                    L = 2 * r

                    # Characteristic flow rate: you can take, for example, dataset.mousev
                    U = dataset.mousev

                    # Calculate Struhal number: St = f * L / U
                    St = dominant_freq * L / U

                    print(f"Calculated Struhal number: {St:.4f} (f = {dominant_freq:.4f} Hz, L = {L}, U = {U})")

                    # -------------------------------------------------------------------------------------------------

                    # Retrieve cylinder parameters (assuming the environment type is “magnus” or “DFG_benchmark”)
                    # radius = dataset.env_info[0].get("r", 10)  # if r is not specified, the default value is used
                    # center = np.array([dataset.env_info[0].get("x", p.shape[1] // 2),
                    #                    dataset.env_info[0].get("y", p.shape[0] // 2)])
                    # # Bring center to full-resolution (since p_img is cut with resolution_factor indentation)
                    # center_full = center * resolution_factor
                    #
                    # # Set the set of angles from 0 to π (0 is the stagnation point, π is the opposite)
                    # num_points = 100  # number of points along the arc
                    # phi = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # angles in radians
                    #
                    # # Calculate the coordinates of the points on the circle
                    # # If radius is given in units of the original grid, then for full-res we scale it by multiplying it by resolution_factor
                    # x_coords = center_full[0] + radius * resolution_factor * np.cos(phi)
                    # y_coords = center_full[1] + radius * resolution_factor * np.sin(phi)
                    #
                    # # Since p_img is obtained after resolution_factor indentation, shift coordinates
                    # x_coords_adj = x_coords - resolution_factor
                    # y_coords_adj = y_coords - resolution_factor
                    #
                    # # Ensure that the coordinates remain within the p_img array
                    # x_coords_adj = np.clip(x_coords_adj, 0, p.shape[1] - 1)
                    # y_coords_adj = np.clip(y_coords_adj, 0, p.shape[0] - 1)
                    #
                    # # Extract pressure values (by nearest pixel)
                    # p_values = [p[int(round(y)), int(round(x))] for x, y in zip(x_coords_adj, y_coords_adj)]

                    # Plot the pressure versus angle (convert radians to degrees)
                    # plt.plot(6, figsize=(8, 4))
                    # plt.plot(np.degrees(phi), p_values, 'bo-', label='Pressure')
                    # plt.xlabel('Angle (degrees)')
                    # plt.ylabel('Pressure')
                    # plt.title('Pressure distribution on the front half of the cylinder')
                    # plt.grid(True)
                    # plt.legend()
                    # plt.show()

                plt.show()

            if key==ord('n'):
                break

            if key==ord('q'):
                exit_loop = True
                break

            print(f"FPS: {last_FPS}")
            FPS += 1
            if time.time()-last_time>=1:
                last_time = time.time()
                last_FPS=FPS
                FPS = 0

if save_movie:
    movie_p.release()
    movie_v.release()
    movie_a.release()

