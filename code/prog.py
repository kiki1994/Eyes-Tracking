# -*- coding: utf-8 -*-

import argparse

person_set = ["P%02d" % i for i in range(15)]
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--person', choices=person_set, default='P00', help='select test set')
parser.add_argument('-g', '--gpu', choices=['%d' % i for i in range(4)], default='0', help="choose a gpu")
args = parser.parse_args()
person_num = args.person