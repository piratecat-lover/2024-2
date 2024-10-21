import torch


def gru_forward(
    input: torch.Tensor,  # (batch_size, input_size), dtype=torch.double
    hidden: torch.Tensor,  # (batch_size, hidden_size), dtype=torch.double
    weight_ih: torch.Tensor,  # (3 * hidden_size, input_size), dtype=torch.double
    weight_hh: torch.Tensor,  # (3 * hidden_size, hidden_size), dtype=torch.double
    bias_ih: torch.Tensor,  # (3 * hidden_size), dtype=torch.double
    bias_hh: torch.Tensor,  # (3 * hidden_size), dtype=torch.double
):
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################

    ##############################################################################
    #                          END OF YOUR CODE                                  #
    ##############################################################################
    output: torch.Tensor  # (batch_size, hidden_size)
    return output


def gru_backward(
    grad_output: torch.Tensor,  # (batch_size, hidden_size), dtype=torch.double
    #
    input: torch.Tensor,  # (batch_size, input_size), dtype=torch.double
    hidden: torch.Tensor,  # (batch_size, hidden_size), dtype=torch.double
    weight_ih: torch.Tensor,  # (3 * hidden_size, input_size), dtype=torch.double,
    weight_hh: torch.Tensor,  # (3 * hidden_size, hidden_size), dtype=torch.double
    bias_ih: torch.Tensor,  # (3 * hidden_size), dtype=torch.double
    bias_hh: torch.Tensor,  # (3 * hidden_size), dtype=torch.double
    # IMPORTANT!
    # Thhe order of weight_ih, weight_hh, bias_ih, bias_hh (3 hidden_size, input_size)
    # is reset, update, new (current)"
):
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################

    ##############################################################################
    #                          END OF YOUR CODE                                  #
    ##############################################################################
    grad_hidden: torch.Tensor  # (batch_size, hidden_size)
    grad_weight_ih: torch.Tensor  # (3 * hidden_size, input_size)
    grad_weight_hh: torch.Tensor  # (3 * hidden_size, hidden_size)
    grad_bias_ih: torch.Tensor  # (3 * hidden_size)
    grad_bias_hh: torch.Tensor  # (3 * hidden_size)
    return grad_hidden, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
