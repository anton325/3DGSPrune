import torch


def build_covariance_from_scaling_rotation(scaling, rotation):
    L = build_scaling_rotation(scaling, rotation)
    
    # cov = RSS.TR.T
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_lowerdiag(actual_covariance)
    return symm

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), device = L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty

def build_rotation(r):
    """
    quaternion to rotation matrix
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.shape[0], 3, 3), device = r.device)

    r = q[:, 0] # real part
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
    
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


if __name__ == "__main__":
    scaling = torch.ones((2000,3))
    rotations = torch.ones((2000,4))
    build_covariance_from_scaling_rotation(scaling, rotations)