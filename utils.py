from collections import deque
import torch

def reconstruct_codec(flattened):

    flattened = deque(flattened) # makes it efficient
    lists = [[],[],[]]

    while len(flattened)>=7:
        lists[0] += [flattened.popleft()]
        lists[1] += [flattened.popleft()]
        lists[2] += [flattened.popleft()]
        lists[2] += [flattened.popleft()]
        lists[1] += [flattened.popleft()]
        lists[2] += [flattened.popleft()]
        lists[2] += [flattened.popleft()]

    if any([v < 4 or v > 4094 for v in lists[0]]) or any([v > 4095 for v in lists[1]+lists[2]]):
        raise Exception("Invalid Codec") 

    return [torch.tensor(l, dtype=torch.int).unsqueeze(0).to("cuda") for l in lists]

def flat_codec(codec):

    flattened = []
    for i in range(len(codec[0][0])):
        flattened.append(codec[0][0][i])
        flattened.append(codec[1][0][2*i])
        flattened.append(codec[2][0][4*i])

        if 4*i + 1 < len(codec[2][0]):
            flattened.append(codec[2][0][4*i + 1])

        if 2*i + 1 < len(codec[1][0]):
            flattened.append(codec[1][0][2*i + 1])
            flattened.append(codec[2][0][4*i + 2])

            if 4*i + 3 < len(codec[2][0]):
                flattened.append(codec[2][0][4*i + 3])

    return flattened