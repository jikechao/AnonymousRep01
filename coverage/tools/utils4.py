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
        inputs = inputs[:600]
        for idx, x in enumerate(inputs):
            #shape of x is 28 * 28
            x = np.expand_dims(x, axis=0)
            # start_time = datetime.datetime.now()
            self.update_coverage(input_data=x, threshold=threshold,K=K_value,b = b)
            activate_neurons, total_neurons, rate, rate_1, rate_2, rate_3 = self.neuron_covered()
            if idx == 100:
                rate_100 = rate
                rate_1_100 = rate_1
                rate_2_100 = rate_2
                rate_3_100 = rate_3
            elif idx == 500:
                rate_500 = rate
                rate_1_500 = rate_1
                rate_2_500 = rate_2
                rate_3_500 = rate_3
            # end = (datetime.datetime.now()-start_time)
            # print(end)
            if rate != cur_percent:
                cur_percent = rate
                # print(idx, activate_neurons, total_neurons, cur_percent)
        print("神经元覆盖率:%.8f----边界神经元覆盖率:%.8f----强神经元覆盖率:%.8f----TOPK覆盖率%.8f"%(rate,rate_1,rate_2,rate_3))
        return rate_100, rate_1_100, rate_2_100, rate_3_100, rate_500, rate_1_500, rate_2_500, rate_3_500, rate,rate_1,rate_2,rate_3


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
        layers = [layer for layer in self.model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
        input_data = kwargs['input_data']
        b = kwargs['b']
        k_section = kwargs['k_section']
        neuron_list_dict = kwargs['neuron_list_dict']
        cover_section_num = 0
        cover_section_num_100 = 0
        cover_section_num_500 = 0
        error = 0
        #section_dict =defaultdict(bool)
        for n,data in enumerate(input_data):
            data = np.expand_dims(data, axis=0)
            intermediate_layer_outputs = self.intermediate_layer_model.predict(data)
            start3 = datetime.datetime.now()
            for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                scaled = self.scale(intermediate_layer_output[0])
                for num_neuron in range(scaled.shape[-1]):
                    d = (b[(i, num_neuron)][1] - b[(i, num_neuron)][0]) / k_section
                    scaled_output = scaled[..., num_neuron]
                    mean = np.mean(scaled_output)
                    if mean > b[(i, num_neuron)][1] or mean < b[(i, num_neuron)][0]:
                        continue
                    if mean == b[(i, num_neuron)][1] and not neuron_list_dict[(i, num_neuron)][-1]:
                        neuron_list_dict[(i, num_neuron)][-1] = True
                        #singel_cover_section_num += 1
                        continue

                    try:
                        sec_id = int(np.floor((mean - b[(i, num_neuron)][0]) / d))
                        if sec_id == k_section:
                            continue
                        if not neuron_list_dict[(i, num_neuron)][sec_id]:
                            neuron_list_dict[(i, num_neuron)][sec_id] = True
                            # singel_cover_section_num += 1
                    except ValueError:
                        error += 1
                        continue
            if n == 100:
                for i, layer in enumerate(layers):
                    for index in range(layer.output_shape[-1]):
                        for section_id in range(k_section):
                            if neuron_list_dict[(i, index)][section_id]:
                                cover_section_num_100 += 1
            elif n == 500:
                for i, layer in enumerate(layers):
                    for index in range(layer.output_shape[-1]):
                        for section_id in range(k_section):
                            if neuron_list_dict[(i, index)][section_id]:
                                cover_section_num_500 += 1

            # elapsed5 = (datetime.datetime.now() - start3)
            # print("single picture Time used: ", elapsed5)
            # print("------------")


        for i, layer in enumerate(layers):
            for index in range(layer.output_shape[-1]):
                for section_id in range(k_section):
                    if neuron_list_dict[(i, index)][section_id]:
                        cover_section_num += 1

        print("error-------", error)
        #print(cover_section_num)
        return cover_section_num_100,cover_section_num_500,cover_section_num



    def final_coverage(self, **kwargs):
        cov_name = self.__class__.__name__
        print("Coverage Type: {}".format(cov_name))
        cur_percent = 0.0
        inputs = kwargs['inputs']
        k_section = kwargs['k_section']
        model = self.model
        #b为边界字典
        b = kwargs['b']
        layers = [layer for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
        neuron_list_dict = defaultdict(list)
        for i, layer in enumerate(layers):
            for index in range(layer.output_shape[-1]):
                neuron_list_dict[(i, index)] = [False for _ in  range(k_section)]

        cover_section_num = self.update_coverage(input_data=inputs, k_section=k_section,b=b,neuron_list_dict = neuron_list_dict)
        print(len(self.model_layer_dict))
        #rate = [csn  / (len(self.model_layer_dict) * k_section) for csn in cover_section_num]
        #rate = cover_section_num / len(self.model_layer_dict) * k_section
        #print(cover_section_num)
        rate = [coverage / (len(self.model_layer_dict) * k_section) for  coverage in cover_section_num]
        print("kmnc_100_500_1000:", rate)
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

