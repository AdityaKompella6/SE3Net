import torch
from torch.autograd import Function

eps = 1e-12


def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    vec = vector.contiguous().view(N, 3)
    output = vec.new().resize_(N, 3, 3).fill_(0)
    output[:, 0, 1] = -vec[:, 2]
    output[:, 1, 0] = vec[:, 2]
    output[:, 0, 2] = vec[:, 1]
    output[:, 2, 0] = -vec[:, 1]
    output[:, 1, 2] = -vec[:, 0]
    output[:, 2, 1] = vec[:, 0]
    return output


def create_rot_from_aa(params):
    # Get the un-normalized axis and angle
    N = params.size(0)
    axis = params.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute sines
    S = torch.sin(angle) / angle
    S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

    # Compute cosines
    C = (1 - torch.cos(angle)) / angle2
    C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    rot = (
        torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(params)
    )  # R = I (avoid use expand as it does not allocate new memory)
    rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
    rot += K2 * C.expand(
        N, 3, 3
    )  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2
    return rot


class SE3ToRtFunction(torch.autograd.Function):
    # def __init__(self):
    #     self.num_cols = 4

    @staticmethod
    def forward(ctx, axis_angles):
        batch_size, num_se3, num_params = axis_angles.size()
        tot_se3 = batch_size * num_se3
        num_cols = 4
        T = axis_angles[:,:,:3].unsqueeze(-1)
        temp = torch.zeros((batch_size, num_se3, 3, num_cols-1),device=T.device)
        output = torch.cat((temp,T),dim = 3)
        # print(axis_angles.new().resize_(batch_size, num_se3, 3, num_cols))
        outputv = output.view(tot_se3, 3, num_cols)
        params = axis_angles.view(tot_se3, -1)  # Bk x num_params
        rot_dim = 3
        rot_params = params.narrow(1, 3, rot_dim)
        outputv.narrow(2, 0, 3).copy_(create_rot_from_aa(rot_params))
        ctx.save_for_backward(axis_angles, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        batch_size, num_se3, num_params = input.size()
        tot_se3 = batch_size * num_se3
        rot_dim = 3  # Number of rotation parameters

        # Init memory for grad input
        grad_input = input.new().resize_as_(input)
        grad_input_v = grad_input.view(tot_se3, -1)  # View it with (Bk) x num_params

        # Get grad output & input in correct shape
        num_cols = 4
        grad_output_v = grad_output.view(tot_se3, 3, num_cols)  # (Bk) x 3 x num_cols
        params = input.view(tot_se3, -1)  # (Bk) x num_params
        rot_params = params.narrow(
            1, 3, rot_dim
        )  # (Bk) x rotdim (Last few parameters are the rotation parameters)
        grad_rot_params = grad_input_v.narrow(
            1, 3, rot_dim
        )  # (Bk) x rotdim (Last few parameters are the rotation parameters)
        rot = output.view(tot_se3, 3, num_cols).narrow(
            2, 0, 3
        )  # (Bk) x 3 x 3 => 3x3 rotation matrix
        axis = params.view(tot_se3, -1, 1).narrow(
            1, 3, rot_dim
        )  # (Bk) x 3 x 1 => 3 unnormalized AA parameters (note that first 3 parameters are [t])
        angle2 = (axis * axis).sum(
            1
        )  # (Bk) x 1 x 1 => Norm of the vector (squared angle)
        nSmall = angle2.lt(eps).sum()  # Num angles less than threshold

        # Compute: v x (Id - R) for all the columns of (Id-R)
        I = (
            torch.eye(3).type_as(rot).repeat(tot_se3, 1, 1).add(rot, alpha=-1)
        )  # (Bk) x 3 x 3 => Id - R
        vI = torch.cross(axis.expand_as(I), I, 1)  # (Bk) x 3 x 3 => v x (Id - R)

        # Compute [v * v' + v x (Id - R)] / ||v||^2
        vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
        vV = (vV + vI) / (
            angle2.view(tot_se3, 1, 1).expand_as(vV)
        )  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

        # Iterate over the 3-axis angle parameters to compute their gradients
        # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
        for k in range(3):
            # Create skew symmetric matrix
            skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

            # For those AAs with angle^2 < threshold, gradient is different
            # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
            if nSmall > 0:
                vec = torch.zeros(1, 3).type_as(skewsym)
                vec[0][k] = 1  # Unit vector
                idskewsym = create_skew_symmetric_matrix(vec)
                for i in range(tot_se3):
                    if angle2[i].squeeze()[0] < eps:
                        skewsym[i].copy_(
                            idskewsym.squeeze()
                        )  # Use the new skew sym matrix (around identity)

            # Compute the gradients now
            out = (
                (torch.bmm(skewsym, rot) * grad_output_v.narrow(2, 0, 3)).sum(2).sum(1)
            )  # [(Bk) x 1 x 1] => (vV x R) .* gradOutput
            grad_rot_params[:, k] = out
        grad_input_v[:, 0:3] = grad_output_v[:, :, 3]
        return grad_input


class SE3ToRt(torch.nn.Module):
    def __init__(self):
        super(SE3ToRt, self).__init__()

    def forward(self, input):
        return SE3ToRtFunction().apply(input)

