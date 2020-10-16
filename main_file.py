from class_definations import vision_demo_class
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--timer_val', help='Set the countdown for saving image', type=int, default=5)
parser.add_argument('-s', '--set_number', help='Set folder number for dataset', type=int, default=1)
args = parser.parse_args()


o = vision_demo_class(args.timer_val, args.set_number)

while True:
	o.contact_less_new_GUI()