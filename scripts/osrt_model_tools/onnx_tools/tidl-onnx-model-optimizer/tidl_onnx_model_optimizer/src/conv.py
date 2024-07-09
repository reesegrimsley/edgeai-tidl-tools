# Copyright (c) {2023 - 2024} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
"""Attention block detection and optimization"""

import logging
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import onnx_graphsurgeon as gs
import onnx
import copy

from .common import bordered
from .common import find_in_layers, find_node_idx, find_out_layers, is_ancestor
from .common import find_in_layer, find_out_layer
from .common import remove_node

def tidl_modify_conv(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Wrapper function to modify unsupported convolution layer configurtions 
    """
    tidl_convert_conv_even_filter_to_odd(graph, onnx_graph)


def tidl_convert_conv_even_filter_to_odd(graph: gs.Graph, onnx_graph: onnx.GraphProto, zero_points={'Conv_Name_Fake_Example': -0.001}):
    '''
    Even-sized convolution kernels are not supported in TIDL
    Replace even-sized kernels with next-size up odd kernels, with padding handled appropriately. Additional filter weights are the zero_points

    :param zero_points: On a per-layer basis, the zero-point for asymmetric quantization. This is a dictionary where key is the layer name, and value is the zero-point for that layer (assumed same for all layers, i.e. no grouping)
    
    Some tricks are required here due to Conv layer implementation in TIDL being 'SAME' only. This requires padding be handled outside the layer itself (due to asymmetric pads). Asymmetric quantization is not well supported for these layers, since the zero-point is unknown until calibration. The zero-point fills the additional convolution weights
    '''
    #identify conv nodes
    #find conv nodes w/ even sized kernels
    #replace even sized kernel with odd, and move values into appropriate shape Constant tensor
    #reset pad values in conv node
    #create Pad node that handles all padding, include 'zero point' values?
    #make Conv input the Pad input, Pad output the Conv input

    conv_nodes = [node for node in graph.nodes if node.op == "Conv"]

    for conv in conv_nodes:
        kernel_shape = conv.attrs['kernel_shape']
        pads = conv.attrs['pads']
        weight_tensor = conv.inputs[1]

        conv_input = conv.inputs[0]

        MAX_SUPPORTED_CONV_KERNEL = 7 #7x7 is largest validated layer size
        if kernel_shape[0] % 2 == 0 and kernel_shape[0] < MAX_SUPPORTED_CONV_KERNEL and kernel_shape[1] == kernel_shape[0]:
            print('Promoting conv node (%s) size (%d x %d) to next size up' % (conv.name, kernel_shape[0], kernel_shape[1]))

            new_size = kernel_shape[0] + 1
            new_shape = [new_size, new_size]

            zero_p = zero_points.get(conv.name, 0)
            
            new_weights_shape = [*weight_tensor.shape[:2], *new_shape]

            # is it correct to put the zero point here or only in the layer padding
            new_weights = np.full(new_weights_shape, zero_p, dtype=np.float32)
            # We will pad left and top side of the filter weights with the fill_value / zero-point as we increase the spatial dimensions by 1
            new_weights[:,:,1:,1:] = weight_tensor.values

            new_weights_tensor = gs.Constant(weight_tensor.name, new_weights)
            print(new_weights_tensor.values.shape)
            conv.inputs[1] = new_weights_tensor


            conv.attrs['kernel_shape'] = new_shape
            print('  New conv kernel shape: ')


            pad_name = 'Pad/' + conv.name

            pads = copy.copy(pads)
            pads[0] += 1 # x1 (height) +1  to account for larger filter
            pads[1] += 1 # x2 (width) +1 to account for larger filter
            all_pads = np.asarray([0,0, pads[0], pads[1], 0, 0, pads[2], pads[3] ]) #incorporate all dimensions: depending on opset, may not support axis specification 
            pads_tensor = gs.Constant(pad_name + '_pads', np.asarray(all_pads, np.int64))
            fill_value_tensor = gs.Constant(pad_name + '_fill', np.asanyarray([zero_p], dtype=np.float32))


            conv.attrs['pads'] = [0,0,0,0]

            pad_attrs = {
                'mode' : 'constant'
            }
            pad_inputs = [conv_input, pads_tensor, fill_value_tensor]
            pad_outputs = [gs.Variable(pad_name+'_output', dtype=conv_input.dtype)]

            print('  Adding Pad layer with dimensions (%d,%d,%d,%d) and resetting conv pads to 0\'s' % (pads[0], pads[1], pads[2], pads[3]))

            pad_node = gs.Node('Pad', pad_name, pad_attrs, pad_inputs, pad_outputs)

            conv.inputs[0] = pad_outputs[0]
            graph.nodes.append(pad_node)

