import os

from reproduce.cifar_jobs import jobs_with_augmentation, jobs_without_augmentation

if __name__ == '__main__':
    jobs = jobs_with_augmentation + jobs_without_augmentation
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        os.system(job)
