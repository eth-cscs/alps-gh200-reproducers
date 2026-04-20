module pdsygst_tests
   use iso_fortran_env, only: error_unit, sp => real32, dp => real64
   use mpi_f08

   implicit none

   external blacs_pinfo
   external blacs_get
   external blacs_gridinit
   external blacs_gridinfo
   external blacs_gridexit
   external blacs_exit
   integer, external :: numroc

   public :: pdsygst_test

contains
   subroutine terminate(ictxt)
      integer, intent(in), optional :: ictxt
      integer :: ierr

      if (present(ictxt)) call blacs_gridexit(ictxt)
      call blacs_exit(1)
      call mpi_finalize(ierr)
      error stop 1
   end subroutine terminate

   subroutine setup_mpi(nprow, npcol, rank, nprocs)
      integer, intent(in) :: nprow, npcol
      integer, intent(out) :: rank, nprocs
      integer:: ierr

      ! don't use threading support in MPI, in order to simplify analysis.
      call mpi_init(ierr)

      call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
      call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)

      if (nprocs /= nprow*npcol) then
         if (rank == 0) then
            write (error_unit, *) 'ERROR: The test suite needs to run with exactly ', nprow*npcol, ' processes'
            write (error_unit, *) 'ERROR: Got ', nprocs, ' processes'
         end if
         call terminate()
      end if
   end subroutine setup_mpi

   subroutine teardown_mpi()
      integer:: ierr

      call mpi_finalize(ierr)

   end subroutine teardown_mpi

   subroutine set_random_matrix_d(a, s)

      real(kind=dp), dimension(:, :), intent(out) :: a

      real(kind=dp), dimension(:, :), allocatable :: rand
      integer :: n, i

      integer, allocatable:: seed(:)
      integer :: nseed
      integer, intent(in), optional :: s

      call random_seed(size=nseed)
      allocate (seed(nseed))
      if (present(s)) then
         seed(:) = s
      else
         seed(:) = 0
      end if
      call random_seed(put=seed)
      deallocate (seed)

      if (size(a, 1) /= size(a, 2)) then
         write (error_unit, *) 'ERROR: Matrix must be square'
         call terminate()
      end if

      n = size(a, 1)

      allocate (rand(n, n))

      call random_number(rand)
      a = rand

      deallocate (rand)

      ! Make hermitian
      a = matmul(a, transpose(a))

      ! Make positive definite
      do i = 1, n
         a(i, i) = a(i, i) + n
      end do

   end subroutine set_random_matrix_d

   subroutine init_desc(desc)
      integer, intent(out), dimension(9) :: desc

      desc(:) = 0
      desc(2) = -1
   end subroutine init_desc

   subroutine pdsygst_test

      integer, parameter :: n = 7296

      integer:: nprow, npcol
      integer :: rank, numprocs, myrow, mycol
      integer :: sctxt, ictxt, ictxt_0
      integer :: info, lld, nb, ma, na
      integer :: desca(9), desca_local(9)
      integer :: descb(9), descb_local(9)
      real(kind=dp), dimension(:, :), allocatable :: A, A_local
      real(kind=dp), dimension(:, :), allocatable :: B, B_local
      character :: uplo = 'U'
      real(kind=dp) :: s
      integer :: start_count, stop_count, rate
      real(kind=dp) :: total_time, avg_time, max_time, min_time

      nprow = 8
      npcol = 4
      nb = 64

      call setup_mpi(nprow, npcol, rank, numprocs)

      ! Setup BLACS
      call blacs_get(0, 0, sctxt)
      ictxt_0 = sctxt
      call blacs_gridinit(ictxt_0, 'R', 1, 1)
      call blacs_get(0, 0, sctxt)
      ictxt = sctxt
      call blacs_gridinit(ictxt, 'R', nprow, npcol)
      call blacs_pinfo(rank, numprocs)
      call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)

      ! Setup full matrices on rank 0
      call init_desc(desca)
      call init_desc(descb)
      if (rank == 0) then
         allocate (A(n, n))
         allocate (B(n, n))

         call descinit(desca, n, n, n, n, 0, 0, ictxt_0, n, info)
         call descinit(descb, n, n, n, n, 0, 0, ictxt_0, n, info)

         call set_random_matrix_d(A)
         call set_random_matrix_d(B)
      end if

      ! Allocate local matrices
      ma = numroc(n, nb, myrow, 0, nprow)
      na = numroc(n, nb, mycol, 0, npcol)
      lld = max(1, ma)
      allocate (A_local(ma, na))
      allocate (B_local(ma, na))

      ! + --------- +
      ! | ScaLAPACK |
      ! + --------- +

      ! Distribute full matrix to ranks
      call descinit(desca_local, n, n, nb, nb, 0, 0, ictxt, lld, info)
      call descinit(descb_local, n, n, nb, nb, 0, 0, ictxt, lld, info)
      call mpi_barrier(MPI_COMM_WORLD, info)
      call system_clock(start_count, rate)
      call pdgemr2d(n, n, A, 1, 1, desca, A_local, 1, 1, desca_local, ictxt)
      call pdgemr2d(n, n, B, 1, 1, descb, B_local, 1, 1, descb_local, ictxt)
      call system_clock(stop_count)
      call mpi_barrier(MPI_COMM_WORLD, info)

      total_time = real(stop_count - start_count) / real(rate)
      call mpi_allreduce(total_time, avg_time, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, max_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, min_time, 1, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, info)
      if (rank == 0) then
         write(error_unit,'(A,F6.3,A," [",F6.3,", ",F6.3,"] seconds")') 'pdgemr2d: ', avg_time/numprocs, ', ', min_time, max_time
      end if

      ! Perform cholesky decomposition of B

      call mpi_barrier(MPI_COMM_WORLD, info)
      call system_clock(start_count, rate)
      call pdpotrf(uplo, n, B_local, 1, 1, descb_local, info)
      call system_clock(stop_count)
      call mpi_barrier(MPI_COMM_WORLD, info)

      total_time = real(stop_count - start_count) / real(rate)
      call mpi_allreduce(total_time, avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, info)

      if (rank == 0) then
         write(error_unit,'(A,F6.3,A," [",F6.3,", ",F6.3,"] seconds")') 'pdpotrf : ', avg_time/numprocs, ', ', min_time, max_time
      end if

      if (info /= 0) then
         write (error_unit, *) 'ERROR: pzpotrf returned info = ', info
         call terminate(ictxt)
      end if

      ! Transform to standard eigenvalue problem
      call mpi_barrier(MPI_COMM_WORLD, info)
      call system_clock(start_count, rate)
      call pdsygst(1, uplo, n, A_local, 1, 1, desca_local, B_local, 1, 1, descb_local, s, info)
      call system_clock(stop_count)
      call mpi_barrier(MPI_COMM_WORLD, info)
      if (info /= 0) then
         write (error_unit, *) 'ERROR: pdsygst returned info = ', info
         call terminate(ictxt)
      end if

      total_time = real(stop_count - start_count) / real(rate)
      !write(error_unit,*) 'rank ', rank, ' clock rate = ', rate

      !write (error_unit, *) 'Process ', rank, ': pdsygst execution time: ', total_time, ' seconds'

      !write (error_unit, *) 'pdsygst execution time: ', end_time - start_time, ' seconds'
      call mpi_allreduce(total_time, avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, info)
      call mpi_allreduce(total_time, min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, info)
      if (rank == 0) then
         write(error_unit,'(A,F6.3,A," [",F6.3,", ",F6.3,"] seconds")') 'pdsygst : ', avg_time/numprocs, ', ', min_time, max_time
      end if

      if (rank == 0) then
         if (allocated(A)) deallocate (A)
         if (allocated(B)) deallocate (B)
      end if
      if (allocated(A_local)) deallocate (A_local)
      if (allocated(B_local)) deallocate (B_local)

      call blacs_gridexit(ictxt)
      call blacs_exit(1)
      call teardown_mpi()
   end subroutine pdsygst_test

end module pdsygst_tests

program test_pdsygst
   use pdsygst_tests, only: pdsygst_test

   implicit none

   call pdsygst_test()

end program test_pdsygst
