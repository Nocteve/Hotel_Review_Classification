def get_sample(self, sam_size=1, _except=[]):
    except_set = set(_except)
    output = []
    while len(output) < sam_size:
        x = self.sampler.sample_n(1)
        if int(x[0]) not in except_set:
            output.append(int(x[0]))
    return output