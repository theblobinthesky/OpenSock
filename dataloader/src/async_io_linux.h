//
// Created by workstation on 3/28/25.
//

#ifndef ASYNC_IO_LINUX_H
#define ASYNC_IO_LINUX_H


/*
        io_uring ring;
        off_t insize;
        int ret;

        int infd = open(
            "/home/workstation/Downloads/OpenSock/dataloader/noxfile.py",
            O_RDONLY);
        if (infd < 0) {
            perror("open infile");
            return;
        }

        int outfd = open(
            "/home/workstation/Downloads/OpenSock/dataloader/noxfile.py2",
            O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (outfd < 0) {
            perror("open outfile");
            return;
        }

        if (setup_context(QD, &ring))
            return;

        if (get_file_size(infd, &insize))
            return;

        ret = copy_file(infd, outfd, &ring, insize);

        close(infd);
        close(outfd);
        io_uring_queue_exit(&ring);
*/

#endif //ASYNC_IO_LINUX_H
