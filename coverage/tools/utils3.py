from collections import defaultdict
import random
import numpy as np
from keras.models import Model
from interval import Interval
from time import time
import datetime



class BaseCoverage:

    def __init__(self, model):
        self.model = model

        self.layer_names = [layer.name
                            for layer in self.model.layers
                            if 'flatten' not in layer.name and 'input' not in layer.name]



        self.intermediate_layer_model = Model(inputs=self.model.input,
                                              outputs=[self.model.get_layer(layer_name).output
                                                       for layer_name in self.layer_names])
        self.model_layer_dict = self.init_coverage_tables()
        self.model_layer_dict_1 = self.init_coverage_tables()
        self.model_layer_dict_2 = self.init_coverage_tables()
        self.model_layer_dict_3 = self.init_coverage_tables()
    def init_coverage_tables(self):
        model_layer_dict = defaultdict(bool)
        for layer in self.model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False
        return model_layer_dict

    def neuron_to_cover(self):
        not_covered = [(layer_name, index) for (layer_name, index), v in self.model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self.model_layer_dict.keys())
        return layer_name, index

    def neuron_covered(self):
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        boundry_covered_neurons = len([v for v in self.model_layer_dict_1.values() if v])
        strong_covered_neurons = len([v for v in self.model_layer_dict_2.values() if v])
        topk_covered_neurons = len([v for v in self.model_layer_dict_3.values() if v])
        total_neurons = len(self.model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons),boundry_covered_neurons / (2 * float(total_neurons)),\
               strong_covered_neurons / float(total_neurons),topk_covered_neurons / float(total_neurons)

    def update_coverage(self, **kwargs):
        pass

    def scale(self, intermediate_layer_output, rmax=1, rmin=0):
        """standardized"""
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def final_coverage(self, **kwargs):
        cov_name = self.__class__.__name__
        print("Coverage Type: {}".format(cov_name))
        cur_percent = 0.0
        inputs = kwargs['inputs']
        threshold = kwargs['threshold']
        K_value = kwargs['K']
        b= kwargs['b']
        for idx, x in enumerate(inputs):
            #shape of x is 28 * 28

            x = np.expand_dims(x, axis=0)
            self.update_coverage(input_data=x, threshold=threshold,K=K_value,b = b)
            activate_neurons, total_neurons, rate, rate_1, rate_2, rate_3 = self.neuron_covered()
            if rate != cur_percent:
                cur_percent = rate
                # print(idx, activate_neurons, total_neurons, cur_percent)
        print("神经元覆盖率:%.8f----边界神经元覆盖率:%.8f----强神经元覆盖率:%.8f----TOPK覆盖率%.8f"%(rate,rate_1,rate_2,rate_3))
        return rate,rate_1,rate_2,rate_3


class NeuronCoverage(BaseCoverage):
    def __init__(self, model):
        BaseCoverage.__init__(self, model=model)

    def update_coverage(self, **kwargs):
        input_data = kwargs['input_data']
        threshold = kwargs['threshold']
        b = kwargs['b']
        K_value = kwargs['K']

        # start2 = datetime.datetime.now()
        intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)
        # elapsed2 = (datetime.datetime.now() - start2)
        # print("Prediction Time used: ", elapsed2)
        # print("------------")
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self.scale(intermediate_layer_output[0])
            neurons_list = list()
            for num_neuron in range(scaled.shape[-1]):
                scaled_output = scaled[..., num_neuron]
                mean = np.mean(scaled_output)
                neurons_list.append(mean)
                #NeuronCoverage激活条件
                if mean > threshold and not self.model_layer_dict[(self.layer_names[i], num_neuron)]:
                    self.model_layer_dict[(self.layer_names[i], num_neuron)] = True
                #NeuronBoundaryCoverage激活条件
                if (mean > b[(i,num_neuron)][1] or mean < b[(i,num_neuron)][0]) and not self.model_layer_dict_1[(self.layer_names[i], num_neuron)]:
                    self.model_layer_dict_1[(self.layer_names[i], num_neuron)] = True
                #StrongNeuronCoverage激活条件
                if mean > b[(i,num_neuron)][1] and not self.model_layer_dict_2[(self.layer_names[i], num_neuron)]:
                    self.model_layer_dict_2[(self.layer_names[i], num_neuron)] = True
            #TopKNeuronCoverage激活条件
            neurons_rank = np.argsort(neurons_list)

            for j in range(1, K_value + 1):
                # get top j-th neuron index
                neurons_index = neurons_rank[-j]
                if not self.model_layer_dict_3[(self.layer_names[i], neurons_index)]:
                    self.model_layer_dict_3[(self.layer_names[i], neurons_index)] = True



class K_SectionNeuronCoverage(BaseCoverage):
    def __init__(self, model):
        BaseCoverage.__init__(self, model=model)


    def update_coverage(self, **kwargs):
        input_data = kwargs['input_data']
        b = kwargs['b']
        k_section = kwargs['k_section']
        cover_section_num = 0
        singel_cover_section_num = 0
        # start2 = datetime.datetime.now()
        error = 0
        intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            for num_neuron in range(intermediate_layer_output.shape[-1]):
                d = (b[(i, num_neuron)][1] - b[(i, num_neuron)][0]) / k_section
                #print("神经元%d，%d---覆盖数%d" % (i, num_neuron, singel_cover_section_num))
                cover_section_num = cover_section_num + singel_cover_section_num
                singel_cover_section_num = 0
                section_list = [False for _ in range(k_section)]
                for n in range(intermediate_layer_output.shape[0]):
                    scaled = self.scale(intermediate_layer_output[n])
                    scaled_output = scaled[..., num_neuron]
                    mean = np.mean(scaled_output)
                    if mean > b[(i, num_neuron)][1] or mean < b[(i, num_neuron)][0]:
                        continue
                    if mean == b[(i, num_neuron)][1] and not section_list[-1]:
                        section_list[-1] = True
                        singel_cover_section_num += 1
                        continue
                    try:
                        sec_id = int(np.floor((mean - b[(i, num_neuron)][0]) / d))
                        if sec_id == k_section:
                            continue

                        if not section_list[sec_id]:
                            section_list[sec_id] = True
                            singel_cover_section_num += 1
                    except ValueError:
                        error += 1
                        continue

                # for n in range(intermediate_layer_output.shape[0]):
                #     scaled = self.scale(intermediate_layer_output[n])
                #     scaled_output = scaled[..., num_neuron]
                #     mean = np.mean(scaled_output)
                #     h = len(section_list)
                #     first = 0
                #     last = h - 1
                #     while first <= last:
                #         if mean >section_list[-1][0].upper_bound or mean < section_list[0][0].lower_bound:
                #             break
                #         mid = (last + first) // 2
                #         if section_list[mid][0].lower_bound < mean and section_list[mid][ 0].upper_bound > mean and not section_list[mid][1]:
                #             section_list[mid][1] = True
                #             singel_cover_section_num += 1
                #             break
                #         elif section_list[mid][0].upper_bound < mean:
                #             first = mid + 1
                #         elif section_list[mid][0].lower_bound > mean:
                #             last = mid - 1
                #         else:
                #             break
        # elapsed2 = (datetime.datetime.now() - start2)
        # print("Time used: ", elapsed2)
        # print("------------")
        print("error-------",error)
        return cover_section_num
    def final_coverage(self, **kwargs):
        cov_name = self.__class__.__name__
        print("Coverage Type: {}".format(cov_name))
        cur_percent = 0.0
        inputs = kwargs['inputs']
        k_section = kwargs['k_section']
        #b为边界字典
        b = kwargs['b']
        cover_section_num = self.update_coverage(input_data=inputs, k_section=k_section,b=b)
        print(len(self.model_layer_dict))
        rate = cover_section_num  / (len(self.model_layer_dict) * k_section)
        print("%s : 覆盖率%.8f"%(cov_name , rate))
        return rate

# class NeuronBoundaryCoverage(BaseCoverage):
#     def __init__(self,model):
#         BaseCoverage.__init__(self,model=model)
#
#     def update_coverage(self, **kwargs):
#
#         input_data = kwargs['input_data']
#         b = kwargs['b']
#
#
#         intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)
#         for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
#             # scaled[28,28,6]
#             # print(intermediate_layer_output)
#             scaled = self.scale(intermediate_layer_output[0])
#             for num_neuron in range(scaled.shape[-1]):
#                 # print(num_neuron)
#                 # scaled_output[28, 28]
#                 scaled_output = scaled[..., num_neuron]
#                 mean = np.mean(scaled_output)
#                 if (mean > b[1] or mean <b[0] ) and not self.model_layer_dict[(self.layer_names[i], num_neuron)]:
#                     self.model_layer_dict[(self.layer_names[i], num_neuron)] = True
#
#     def final_coverage(self, **kwargs):
#         cov_name = self.__class__.__name__
#         print("Coverage Type: {}".format(cov_name))
#         cur_percent = 0.0
#         inputs = kwargs['inputs']
#         b = kwargs['b']
#         for idx, x in enumerate(inputs):
#             # shape of x is 28 * 28
#
#             x = np.expand_dims(x, axis=0)
#             self.update_coverage(input_data=x, b=b)
#             activate_neurons, total_neurons, rate = self.neuron_covered()
#             rate = rate / 2
#             if rate != cur_percent:
#                 cur_percent = rate
#                 print(idx, activate_neurons, total_neurons, cur_percent)
#         print(cov_name, cur_percent)
#         return cur_percent


# class StrongNeuronCoverage(BaseCoverage):
#     def __init__(self, model):
#         BaseCoverage.__init__(self, model=model)
#
#     def update_coverage(self, **kwargs):
#
#         input_data = kwargs['input_data']
#         b = kwargs['b']
#
#         intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)
#         for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
#             # scaled[28,28,6]
#             # print(intermediate_layer_output)
#             scaled = self.scale(intermediate_layer_output[0])
#             for num_neuron in range(scaled.shape[-1]):
#                 # print(num_neuron)
#                 # scaled_output[28, 28]
#                 scaled_output = scaled[..., num_neuron]
#                 mean = np.mean(scaled_output)
#                 if mean > b[1] and not self.model_layer_dict[(self.layer_names[i], num_neuron)]:
#                     self.model_layer_dict[(self.layer_names[i], num_neuron)] = True
#
#     def final_coverage(self, **kwargs):
#         cov_name = self.__class__.__name__
#         print("Coverage Type: {}".format(cov_name))
#         cur_percent = 0.0
#         inputs = kwargs['inputs']
#         b = kwargs['b']
#         for idx, x in enumerate(inputs):
#             # shape of x is 28 * 28
#
#             x = np.expand_dims(x, axis=0)
#             self.update_coverage(input_data=x, b=b)
#             activate_neurons, total_neurons, rate = self.neuron_covered()
#             if rate != cur_percent:
#                 cur_percent = rate
#                 print(idx, activate_neurons, total_neurons, cur_percent)
#         print(cov_name, cur_percent)
#         return cur_percent

