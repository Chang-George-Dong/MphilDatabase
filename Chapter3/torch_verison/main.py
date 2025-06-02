from config import *

if __name__ == "__main__":
    if if_Train:
        train()
    if if_Attack:
        perform_attack()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train or Attack')
#     parser.add_argument('command', choices=['train', 'attack'], help='Choose to train or attack')
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if args.command == 'train':
#         train(device)
#     elif args.command == 'attack':
#         perform_attack(device)


# train(DEVICE)
