import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import json

#  True parameters
true_mean1 = np.array([0.0, 0.0])
true_mean2 = np.array([0.0, 0.0])
true_var1 = (np.array([0.3, 0.4]))
true_var2 = (np.array([0.5, 0.3]))
true_cov1 = np.diag(true_var1)
true_cov2 = np.diag(true_var2)

# Simulation parameters
sim_mean1 = np.array([0.0, 0.0])
sim_mean2 = np.array([0.0, 0.0])
var1 = np.array([0.5, 0.6])
var2 = np.array([0.7, 0.5])
sim_var1 = var1
sim_var2 = var2
sim_cov1 = np.diag(sim_var1)
sim_cov2 = np.diag(sim_var2)

# sim_mean1 = np.array([0.7, 0.6])
# sim_mean2 = np.array([0.5, 0.6])
# sim_var1 = np.array([0.1, 0.5])
# sim_var2 = np.array([0.5, 0.4])
# sim_cov1 = np.diag(true_var1)
# sim_cov2 = np.diag(true_var2)

dt = 0.1
sim_time = 20
v = 1.0  # [m/s]
max_range = 20.0  # maximum observation range
state_size = 3  # State size [x,y,yaw]
lm_size = 2  # LM state size [x,y]
n_particle = 10  # number of particles
nth = n_particle / 1.5  # Number of particles for re-sampling
R = np.array([[0.0, 0.0],  # Process Noise
              [0.0, 0.0]])
RFID = np.array([[10.0, -2.0],
                 [15.0, 10.0],
                 [15.0, 17.0],
                 [10.0, 15.0],
                 [30.0, 12.0],
                 [20.0, 20.0],
                 [25.0, 23.0],
                 [-10.0, 15.0]
                 ])
n_landmark = RFID.shape[0]
x_tru = np.array([0, 0, 0])
xEst = np.zeros((state_size, 1))  # SLAM estimation
xDR = np.zeros((state_size, 1))  # Dead reckoning
# history
hxEst = xEst
hxTrue = x_tru
hxDR = x_tru
# create empty lists to store data for statistical analysis
z1_list = []
z2_list = []

show_animation = False


class Particle:

    def __init__(self, n_landmark):
        self.w = 1.0 / n_particle  # particle weight
        self.x = 0.0  # pos-x
        self.y = 0.0  # pos-y
        self.yaw = 45.0
        # landmark x-y positions
        self.lm = np.zeros((n_landmark, lm_size))
        # landmark position covariance
        self.lmP = np.zeros((n_landmark * lm_size, lm_size))


def motion_model(x):
    time = 0
    tru_states = []
    while sim_time >= time:
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        # B = np.array([[dt * math.cos(x[0]), dt * math.cos(x[1])]])   # np.array([0.1, 0.1])
        B = np.array([0.1, 0.1, 0])

        x = F @ x + B
        time = time + dt
        tru_states.append(x)

    return tru_states


def observation(xTrue, rfid):
    # add noise to range observation
    z1 = np.zeros((4, 0))
    z2 = np.zeros((4, 0))
    for i in range(len(rfid[:, 0])):
        dx = rfid[i, 0] - xTrue[0]
        dy = rfid[i, 1] - xTrue[1]
        d = math.hypot(dx, dy)
        if d <= max_range:
            a = np.random.multivariate_normal(true_mean1, true_cov1, 1)
            b = np.random.multivariate_normal(true_mean2, true_cov2, 1)

            # for sensor1
            dxna = dx + a[0][0]
            dyna = dy + a[0][1]
            dna = math.hypot(dxna, dyna)

            # for sensor2
            dxnb = dx + b[0][0]
            dynb = dy + b[0][1]
            dnb = math.hypot(dxnb, dynb)

            z1i = np.array([dna, dxna, dyna, i]).reshape(4, 1)
            z2i = np.array([dnb, dxnb, dynb, i]).reshape(4, 1)
            z1 = np.hstack((z1, z1i))
            z2 = np.hstack((z2, z2i))

    return z1, z2


def get_observation(states):
    z1_list.clear()
    z2_list.clear()
    for i in range(len(states)):
        current_state = states[i]
        observations = observation(current_state, RFID)
        z1 = observations[0]
        z2 = observations[1]
        z1_list.append(z1)
        z2_list.append(z2)
    return


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_jacobians(particle, xf, Pf, Q_cov):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf


def update_kf_with_cholesky(xf, Pf, dv, Q_cov, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q_cov

    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T

    x = xf + W @ dv
    P = Pf - W1 @ W1.T

    return x, P


def add_new_landmark(particle, z, Q_cov):
    r = z[0]
    b = z[1]
    lm_id = int(z[3])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s

    # covariance
    dx = r * c
    dy = r * s
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)
    Gz = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])
    particle.lmP[2 * lm_id:2 * lm_id + 2] = np.linalg.inv(
        Gz) @ Q_cov @ np.linalg.inv(Gz.T)

    return particle


def compute_weight(particle, z, Q_cov):
    lm_id = int(z[3])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singular")
        return 1.0

    try:
        num = math.exp(-0.5 * dx.T @ invS @ dx)
    except OverflowError:
        num = float('inf')
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))

    w = num / den

    return w


def normalize_weight(particles):
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(n_particle):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(n_particle):
            particles[i].w = 1.0 / n_particle

        return particles

    return particles


def update_landmark(particle, z, Q_cov):
    lm_id = int(z[3])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle


def update_with_observation(particles, z, cov):
    for iz in range(len(z[0, :])):

        landmark_id = int(z[3, iz])

        for ip in range(n_particle):
            # new landmark
            if abs(particles[ip].lm[landmark_id, 0]) <= 0.01:
                # print("new")
                particles[ip] = add_new_landmark(particles[ip], z[:, iz], cov)
            # known landmark
            else:
                w = compute_weight(particles[ip], z[:, iz], cov)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], z[:, iz], cov)

    return particles


def resampling(particles):
    """
    low variance re-sampling
    """

    particles = normalize_weight(particles)

    pw = []
    for i in range(n_particle):
        pw.append(particles[i].w)

    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(n_eff)

    if n_eff < nth:  # resampling
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / n_particle) - 1 / n_particle
        resample_id = base + np.random.rand(base.shape[0]) / n_particle

        inds = []
        ind = 0
        for ip in range(n_particle):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)

        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].yaw = tmp_particles[inds[i]].yaw
            particles[i].lm = tmp_particles[inds[i]].lm[:, :]
            particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / n_particle

    return particles


def calc_final_state(particles):
    xEst = np.zeros((state_size, 1))

    particles = normalize_weight(particles)

    for i in range(n_particle):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] = particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    #  print(xEst)

    return xEst


def fast_slam1(particles, z, z2, cov1, cov2):
    particles = predict_particles(particles)

    particles = update_with_observation(particles, z, cov1)

    particles = update_with_observation(particles, z2, cov2)

    particles = resampling(particles)

    return particles


def predict_particles(particles):
    for i in range(n_particle):
        px = np.zeros((state_size, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        u = [0.1, 0.1]
        rnd = np.random.randn(1, 2) @ R
        ud = np.array([0.1, 0.1]) * rnd  # add noise
        particles[i].x = px[0, 0] + u[0]  # + ud[0, 0] + u[0]
        particles[i].y = px[1, 0] + u[1]  # + ud[0, 1] + u[1]

    return particles


def state_to_observation_estimates(x, rfid):
    z1_hat = np.zeros((4, 0))
    for i in range(len(rfid[:, 0])):

        dx = rfid[i, 0] - x[0, 0]
        dy = rfid[i, 1] - x[1, 0]
        d = math.hypot(dx, dy)
        angle = 45  # pi_2_pi(math.atan2(dy, dx) - x[2, 0])
        if d <= max_range:
            zi = np.array([d, dx, dy, i]).reshape(4, 1)
            z1_hat = np.hstack((z1_hat, zi))
    return z1_hat


def compute_params(z_est, z):
    err_x_list = errors(z_est, z)[0]
    mean_x = sum(err_x_list) / len(err_x_list)
    var_x = sum((i - mean_x) ** 2 for i in err_x_list) / len(err_x_list)

    err_y_list = errors(z_est, z)[1]
    mean_y = sum(err_y_list) / len(err_y_list)
    var_y = sum((i - mean_y) ** 2 for i in err_y_list) / len(err_y_list)
    return mean_x, var_x, mean_y, var_y


def errors(z_est, z):  # list of observation estimates(z_est) and a list of actual observations(z)
    err_hat_list = []
    err_x_hat_list = []
    err_y_hat_list = []
    for i in range(len(z_est)):
        if len(z_est[i][0]) == len(z[i][0]):
            # diff = (z_est[i] - z[i]) ** 2
            diff = (z[i] - z_est[i]) ** 2
            err_hat_list.append(diff)  # returns squared error
            err_x_hat_list.extend(diff[0])
            err_y_hat_list.extend(diff[1])
    return err_x_hat_list, err_y_hat_list


def get_neighbours(param, step):
    neighbours = [[(param + step)[0], param[1]],
                  [abs((param - step)[0]), param[1]],
                  [param[0], (param + step)[1]],
                  [param[0], abs((param - step)[1])]]
    # neighbours = [[(param + step)[0],   param[1]],
    #               [(param - step)[0],   param[1]],
    #               [param[0],            (param + step)[1]],
    #               [param[0],            (param - step)[1]],
    #               [(param + step)[0],   (param + step)[1]],
    #               [(param + step)[0],   (param - step)[1]],
    #               [(param - step)[0],   (param - step)[1]],
    #               [(param - step)[0],   (param + step)[1]]]
    # neighbours = [round(num, 1) for num in neighbours]
    return neighbours


def gau_kl(pm, p_v, qm, q_v):
    pv = np.diag(p_v)
    qv = np.diag(q_v)
    # reference:
    # http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
    # returns negative kl divergence in some cases. Not using this function
    if len(qm.shape) == 2:
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = np.linalg.det(pv)  # pv.prod()
    dqv = np.linalg.det(qv)  # qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = np.linalg.inv(qv)  # 1. / qv
    # Difference between means pm, qm
    diff = qm - pm
    # a = (np.transpose(iqv * pv)).sum(0)
    # b = (diff * iqv * diff).sum(0)
    # c = len(pm)
    # d = np.log(dqv) - np.log(dpv)
    # dist = (a + b - c + d).sum(0)
    # return dist
    return (0.5 *
            ((np.log(dqv) - np.log(dpv))  # log |\Sigma_q| / |\Sigma_p|          - maybe use this(log(dqv) - log(dpv))
             + (iqv * pv).sum(axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))  # - N


def kld(pm, p_v, qm, q_v):
    # pv = np.diag(p_v)
    # qv = np.diag(q_v)
    # 2 = p 1 = q
    # f_term = np.log(math.sqrt(p_v)) - np.log(math.sqrt(q_v))
    f_term = np.log(np.sqrt(abs(p_v[0]))) - np.log(np.sqrt(abs(q_v[0]))), np.log(np.sqrt(abs(p_v[1]))) - np.log(
        np.sqrt(abs(q_v[1])))
    if math.isnan(f_term[0]) or math.isnan(f_term[1]):
        print("here nan")

    s_term = (q_v + ((qm - pm) * (qm - pm))) / (2 * p_v)
    t_term = 1 / 2
    distance_log_kl = f_term + s_term - t_term
    if distance_log_kl[0] < 0.0 or distance_log_kl[1] < 0.0:
        exit("Negative value in Kl divergence")
    return distance_log_kl


def gradient(err1_mean, err1_v, err2_mean, err2_v):
    kl1 = kld(np.array(sim_mean1), np.sqrt(abs(np.array(sim_var1))), np.array(err1_mean), np.array(err1_v))
    kl2 = kld(np.array(sim_mean2), np.sqrt(abs(np.array(sim_var2))), np.array(err2_mean), np.array(err2_v))
    dist = (kl1 + kl2).sum(0)
    return dist


def pf():  # pf(sim_mean1, sim_mean2, sim_var1, sim_var2):
    particles_list = []
    z1_hat_list = []
    z2_hat_list = []
    # z1_list = []
    # z2_list = []
    particles = [Particle(n_landmark) for _ in range(n_particle)]
    true_states = motion_model(x_tru)
    get_observation(true_states)
    hxEst = x_tru
    hxTrue = x_tru

    # Run pf and show animation
    for i in range(len(z1_list)):
        z1 = z1_list[i]
        z2 = z2_list[i]
        particles = fast_slam1(particles, z1, z2, sim_cov1, sim_cov2)
        xEst = calc_final_state(particles)
        z1_hat = state_to_observation_estimates(xEst, RFID)
        z2_hat = state_to_observation_estimates(xEst, RFID)
        x_state = xEst[0: state_size]

        # store data history
        hxTrue = np.vstack((hxTrue, true_states[i]))
        hxEst = np.vstack((hxEst, np.transpose(x_state)[0]))
        particles_list.append(particles)
        z1_hat_list.append(z1_hat)
        z2_hat_list.append(z2_hat)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event', lambda event:
                [exit(0) if event.key == 'escape' else None])
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")

            for j in range(n_particle):
                plt.plot(particles[j].x, particles[j].y, ".r")
                plt.plot(particles[j].lm[:, 0], particles[j].lm[:, 1], "xb")

            plt.plot(hxTrue[:, 0], hxTrue[:, 1], "-b")
            plt.plot(hxEst[:, 0], hxEst[:, 1], "-r")
            plt.plot(xEst[0], xEst[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    err1_params = compute_params(z1_hat_list, z1_list)
    err2_params = compute_params(z2_hat_list, z2_list)

    err1_mean = [err1_params[0], err1_params[2]]
    err1_v = [err1_params[1], err1_params[3]]

    err2_mean = [err2_params[0], err2_params[2]]
    err2_v = [err2_params[1], err2_params[3]]

    return err1_mean, err1_v, err2_mean, err2_v


def main():
    global sim_mean1
    global sim_mean2
    global sim_var1
    global sim_var2
    global sim_cov1
    global sim_cov2
    global var1
    global var2
    step = [0.1, 0.1]
    mean_step = [0.01, 0.01]
    # sim_mean1 = np.array([0.1, 0.1])
    # sim_mean2 = np.array([0.1, 0.1])
    # var1 = np.array([0.1, 0.1])
    # var2 = np.array([0.1, 0.1])
    # sim_var1 = np.sqrt(var1)
    # sim_var2 = np.sqrt(var2)
    # sim_cov1 = np.diag(sim_var1)
    # sim_cov2 = np.diag(sim_var2)

    best_mean1 = sim_mean1
    best_mean2 = sim_mean2
    best_var1 = var1
    best_var2 = var2

    # run filter on time here
    err_params = pf()
    err1_mean = err_params[0]
    err1_v = err_params[1]
    err2_mean = err_params[2]
    err2_v = err_params[3]
    best_dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
    print("\nTrue_var1", true_var1, "Predicted_var1:", sim_var1)
    print("True_var2", true_var2, "Predicted_var2:", sim_var2)
    print("Total dist:", best_dist)
    no_change_loop = 0
    loop = 0
    dist_list = []
    dist_list.append(best_dist)
    # temp_dict = {"parameters": param_list}
    # with open('parameter_file.txt', 'w') as f:
    #     json.dump(temp_dict, f)

    while no_change_loop < 20:

        best_was = best_dist
        loop = loop + 1

        neighbours = get_neighbours(var1, np.array(step))
        for i in range(len(neighbours)):
            var1 = neighbours[i]
            sim_var1 = var1
            sim_cov1 = np.diag(sim_var1)
            err_params = pf()
            err1_mean = err_params[0]
            err1_v = err_params[1]
            err2_mean = err_params[2]
            err2_v = err_params[3]
            dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            if dist < best_dist:
                best_dist = dist
                best_var1 = neighbours[i]
                print("\nTrue_var1", true_var1, "Predicted_var1:", sim_var1)
                print("True_var2", true_var2, "Predicted_var2:", sim_var2)
                print("Total dist:", best_dist)
                dist_list.append(best_dist)
                # param_list.append([[sim_mean1, sim_mean2, sim_var1, sim_var2, best_dist]])
        var1 = best_var1
        sim_var1 = var1
        sim_cov1 = np.diag(sim_var1)

        neighbours_2 = get_neighbours(var2, np.array(step))
        for i in range(len(neighbours_2)):
            var2 = neighbours_2[i]
            sim_var2 = var2
            sim_cov2 = np.diag(sim_var2)
            err_params = pf()
            err1_mean = err_params[0]
            err1_v = err_params[1]
            err2_mean = err_params[2]
            err2_v = err_params[3]
            dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            if dist < best_dist:
                best_dist = dist
                best_var2 = neighbours_2[i]
                print("\nTrue_var1", true_var1, "Predicted_var1:", sim_var1)
                print("True_var2", true_var2, "Predicted_var2:", sim_var2)
                print("Total dist:", best_dist)
                dist_list.append(best_dist)
                # param_list.append([[sim_mean1, sim_mean2, sim_var1, sim_var2, best_dist]])
        var2 = best_var2
        sim_var2 = var2
        # sim_cov2 = np.diag(sim_var2)
        sim_cov2 = np.diag(sim_var2)

        neighbours = get_neighbours(sim_mean1, np.array(mean_step))
        for i in range(len(neighbours)):
            sim_mean1 = neighbours[i]
            err_params = pf()
            err1_mean = err_params[0]
            err1_v = err_params[1]
            err2_mean = err_params[2]
            err2_v = err_params[3]
            dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            if dist < best_dist:
                best_dist = dist
                best_mean1 = neighbours[i]
                print("\nTrue_mean1", true_mean1, "predicted_mean1:", sim_mean1)
                print("True_mean2", true_mean2, "predicted_mean2:", sim_mean2)
                print("Total dist:", best_dist)
                dist_list.append(best_dist)
                # param_list.append([[sim_mean1, sim_mean2, sim_var1, sim_var2, best_dist]])
        sim_mean1 = best_mean1

        neighbours = get_neighbours(sim_mean2, np.array(mean_step))
        for i in range(len(neighbours)):
            sim_mean2 = neighbours[i]
            err_params = pf()
            err1_mean = err_params[0]
            err1_v = err_params[1]
            err2_mean = err_params[2]
            err2_v = err_params[3]
            dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            if dist < best_dist:
                best_dist = dist
                best_mean2 = neighbours[i]
                print("\nTrue_mean1", true_mean1, "predicted_mean1:", sim_mean1)
                print("True_mean2", true_mean2, "predicted_mean2:", sim_mean2)
                print("Total dist:", best_dist)
                dist_list.append(best_dist)
                # param_list.append([[sim_mean1, sim_mean2, sim_var1, sim_var2, best_dist]])
        sim_mean2 = best_mean2

        print("\nBest params after iteration:", loop)
        print("True_mean1", true_mean1, "predicted_mean1:", sim_mean1)
        print("True_mean2", true_mean2, "predicted_mean2:", sim_mean2)
        print("True_var1", true_var1, "Predicted_var1:", sim_var1)
        print("True_var2", true_var2, "Predicted_var2:", sim_var2)
        print("Total dist:", best_dist)

        # if best_dist != best_was & no_change_loop == 1:
        #     no_change_loop = 0

        if best_dist == best_was:
            no_change_loop = no_change_loop + 1
            step[0] = step[0] - 0.01
            step[1] = step[1] - 0.01
            if no_change_loop % 5 == 0:
                step = [0.1, 0.1]
            temp_var1 = var1
            temp_var2 = var2
            mu, sigma = 0.0, 0.1
            rand_m = np.random.normal(mu, sigma, 1)
            rand_s = np.random.normal(mu, sigma, 1)
            rand_p = np.array([rand_m[0], rand_s[0]])
            var1 = abs(var1 - np.array([0.1, 0.1]))
            var2 = abs(var2 - np.array([0.1, 0.1]))
            # var1 = abs(sim_var1 - abs(rand_p))
            # var2 = abs(sim_var2 - abs(rand_p))
            sim_var1 = var1
            sim_var2 = var2
            sim_cov1 = np.diag(sim_var1)
            sim_cov2 = np.diag(sim_var2)
            print("\nFound local/global minima.\nChanging step size.\nMaking a random jump")
            # err_params = pf()
            # err1_mean = err_params[0]
            # err1_v = err_params[1]
            # err2_mean = err_params[2]
            # err2_v = err_params[3]
            # best_dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            # print("\nTrue_var1", true_var1, "Predicted_var1:", sim_var1)
            # print("True_var2", true_var2, "Predicted_var2:", sim_var2)
            # print("New Total dist:", best_dist)
            err_params = pf()
            err1_mean = err_params[0]
            err1_v = err_params[1]
            err2_mean = err_params[2]
            err2_v = err_params[3]
            dist = gradient(err1_mean, err1_v, err2_mean, err2_v)
            print("\nTrue_var1", true_var1, "Predicted_var1:", sim_var1)
            print("True_var2", true_var2, "Predicted_var2:", sim_var2)
            print("Total dist:", best_dist)
            if dist < best_dist:
                best_dist = dist

    # kld_list = [el[4] for el in param_list]
    # plt.plot(kld_list)
    # plt.savefig('imgs/kl_hill_' + str(len(param_list)) + '.png')
    plt.plot(dist_list)
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.savefig('imgs/' + "kld_updates" + '.png')

    # with open('filename', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(param_list)
    # with open('params_file.txt', 'w') as f:
    #     for item in param_list:
    #         f.write("%s\n" % item)

    # temp_dict = {"parameters": param_list}
    # with open('parameter_file.txt', 'w') as f:
    #     f.write(temp_dict, f)

    # for d in range(10):
    #     for c in range(10):
    #         for a in range(10):
    #             for b in range(10):
    #                 err_params = pf()
    #                 err1_mean = err_params[0]
    #                 err1_v = err_params[1]
    #                 err2_mean = err_params[2]
    #                 err2_v = err_params[3]
    #                 dis = gradient(err1_mean, err1_v, err2_mean, err2_v)
    #                 sim_mean1 = sim_mean1 + [0.1, 0.1]
    #             sim_mean1 = np.array([0.1, 0.1])
    #             sim_mean2 = sim_mean2 + [0.1, 0.1]
    #         sim_mean2 = np.array([0.1, 0.1])
    #         var1 = var1 + [0.1, 0.1]
    #         sim_var1 = np.sqrt(var1)
    #         sim_cov1 = np.diag(sim_var1)
    #     var1 = [0.1, 0.1]
    #     sim_cov1 = np.diag(sim_var1)
    #     var2 = var2 + [0.1, 0.1]
    #     sim_var2 = np.sqrt(var2)
    #     sim_cov2 = np.diag(sim_var2)


if __name__ == '__main__':
    main()
