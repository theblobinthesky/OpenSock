#if defined(PLATFORM_LINUX)
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <liburing.h>
#include "types.h"

#define QD  2
#define BS (16 * 1024)

struct io_data {
    int read;
    off_t first_offset, offset;
    size_t first_len;
    iovec iov;
};

static int setup_context(unsigned entries, io_uring *ring) {
    int ret = io_uring_queue_init(entries, ring, 0);
    if(ret < 0) {
        fprintf(stderr, "queue_init: %s\n", strerror(-ret));
        return -1;
    }

    return 0;
}

static int get_file_size(int fd, off_t *size) {
    struct stat st;
    if (fstat(fd, &st) < 0 )
        return -1;

    if(S_ISREG(st.st_mode)) {
        *size = st.st_size;
        return 0;
    }

    if (S_ISBLK(st.st_mode)) {
        uint64_t bytes;

        if (ioctl(fd, BLKGETSIZE64, &bytes) != 0)
            return -1;

        *size = bytes;
        return 0;
    }

    return -1;
}

static void queue_prepped(int infd, int outfd, io_uring *ring, io_data *data) {
    io_uring_sqe *sqe = io_uring_get_sqe(ring);
    assert(sqe != null);

    if (data->read) io_uring_prep_readv(sqe, infd, &data->iov, 1, data->offset);
    else io_uring_prep_writev(sqe, outfd, &data->iov, 1, data->offset);

    io_uring_sqe_set_data(sqe, data);
}

static int queue_read(int infd, io_uring *ring, off_t size, off_t offset) {
    io_data *data = (io_data *)malloc(size + sizeof(*data));
    if (!data)
        return 1;

    io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        free(data);
        return 1;
    }

    data->read = 1;
    data->offset = data->first_offset = offset;

    data->iov.iov_base = data + 1;
    data->iov.iov_len = size;
    data->first_len = size;

    io_uring_prep_readv(sqe, infd, &data->iov, 1, offset);
    io_uring_sqe_set_data(sqe, data);
    return 0;
}

static void queue_write(int infd, int outfd, io_uring *ring, io_data *data) {
    data->read = 0;
    data->offset = data->first_offset;

    data->iov.iov_base = data + 1;
    data->iov.iov_len = data->first_len;

    queue_prepped(infd, outfd, ring, data);
    io_uring_submit(ring);
}

int copy_file(int infd, int outfd, io_uring *ring, off_t insize) {
    uint32_t reads, writes;
    io_uring_cqe *cqe;
    off_t write_left, offset;
    int ret;

    write_left = insize;
    writes = reads = offset = 0;

    while (insize > 0 || write_left > 0) {
        int had_reads = reads;
        while (insize > 0 && reads + writes < QD) {
            off_t read_size = std::min(insize, static_cast<off_t>(BS));
            if (queue_read(infd, ring, read_size, offset))
                break;

            insize -= read_size;
            offset += read_size;
            reads++;
        }

        if (had_reads != reads) {
            ret = io_uring_submit(ring);
            if (ret < 0) {
                fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
                break;
            }
        }

        /* Queue is full at this point. Let's find at least one completion */
        int got_comp = 0;
        while (write_left > 0) {
            if (!got_comp) {
                ret = io_uring_wait_cqe(ring, &cqe);
                got_comp = 1;
            } else {
                ret = io_uring_peek_cqe(ring, &cqe);
                if (ret == -EAGAIN) {
                    cqe = NULL;
                    ret = 0;
                }
            }
            if (ret < 0) {
                fprintf(stderr, "io_uring_peek_cqe: %s\n",
                        strerror(-ret));
                return 1;
            }
            if (!cqe)
                break;

            io_data *data = (io_data *)io_uring_cqe_get_data(cqe);
            if (cqe->res < 0) {
                if (cqe->res == -EAGAIN) {
                    queue_prepped(infd, outfd, ring, data);
                    io_uring_cqe_seen(ring, cqe);
                    continue;
                }
                fprintf(stderr, "cqe failed: %s\n",
                        strerror(-cqe->res));
                return 1;
            }

            if (cqe->res != data->iov.iov_len) {
                /* short read/write; adjust and requeue */
                char **iov_base = (char **)&data->iov.iov_base;
                *iov_base += cqe->res;
                data->iov.iov_len -= cqe->res;
                queue_prepped(infd, outfd, ring, data);
                io_uring_cqe_seen(ring, cqe);
                continue;
            }

            /*
             * All done. If write, nothing else to do. If read,
             * queue up corresponding write.
             * */

            if (data->read) {
                queue_write(infd, outfd, ring, data);
                write_left -= data->first_len;
                reads--;
                writes++;
            } else {
                free(data);
                writes--;
            }
            io_uring_cqe_seen(ring, cqe);
        }
    }

    return 0;
}

#endif