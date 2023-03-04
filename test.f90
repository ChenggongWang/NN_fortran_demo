module test_mod
implicit none
! type for NN layer
public NN_Linear_layer_type
type :: NN_Linear_layer_type
    real, dimension(:,:), pointer :: weight=>NULL()
    real, dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
! type for NN structure
public NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

type(NN_FC_type)   :: Rad_NN_FC

contains
! initialize NN 
subroutine nn_init(num_layers, num_hid_nodes)
    integer, intent(in) :: num_layers, num_hid_nodes
    integer :: ilayer, i, j
    Rad_NN_FC%num_layers = num_layers
    Rad_NN_FC%num_hid_nodes = num_hid_nodes
    allocate(Rad_NN_FC%Layers(num_layers))
    ! init each lay
    call random_seed()
    !first layer 102
    ilayer = 1
    allocate(Rad_NN_FC%Layers(ilayer)%weight(102,num_hid_nodes))
    allocate(Rad_NN_FC%Layers(ilayer)%bias(num_hid_nodes))
    call random_number(Rad_NN_FC%Layers(ilayer)%weight)
    call random_number(Rad_NN_FC%Layers(ilayer)%bias)
    do ilayer = 2, num_layers-1
        allocate(Rad_NN_FC%Layers(ilayer)%weight(num_hid_nodes,num_hid_nodes))
        allocate(Rad_NN_FC%Layers(ilayer)%bias(num_hid_nodes))
        call random_number(Rad_NN_FC%Layers(ilayer)%weight)
        call random_number(Rad_NN_FC%Layers(ilayer)%bias)
        Rad_NN_FC%Layers(ilayer)%weight = Rad_NN_FC%Layers(ilayer)%weight - 0.5
        Rad_NN_FC%Layers(ilayer)%bias = Rad_NN_FC%Layers(ilayer)%bias - 0.5
    end do
    !last layer 36
    allocate(Rad_NN_FC%Layers(ilayer)%weight(num_hid_nodes,36))
    allocate(Rad_NN_FC%Layers(ilayer)%bias(36))
    call random_number(Rad_NN_FC%Layers(ilayer)%weight)
    call random_number(Rad_NN_FC%Layers(ilayer)%bias)
end subroutine nn_init

subroutine nn_pred_1d_matmul(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    real, dimension(:), allocatable :: interm1, interm2
    integer::ilayer
    allocate(interm1(size(x)))
    interm1 = x
    ! num_layers matmul called
    do ilayer = 1, Rad_NN_FC%num_layers
        allocate(interm2(size(Rad_NN_FC%Layers(ilayer)%bias)))
        interm2 = matmul(interm1,Rad_NN_FC%Layers(ilayer)%weight) + Rad_NN_FC%Layers(ilayer)%bias
        interm2 = NN_activ(interm2)
        deallocate(interm1)
        allocate(interm1(size(interm2)))
        interm1 = interm2
        deallocate(interm2)
    end do
    y = interm1
    deallocate(interm1)
end subroutine nn_pred_1d_matmul
! run NN with one input (one column)
subroutine nn_pred_1d_sgemm(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    ! local
    integer :: ilayer
    real, dimension(:), allocatable :: interm1, interm2
    ! for sgemm
    integer :: m, k, n, i, j
    
    allocate(interm1(size(x)))
    interm1 = x
    do ilayer = 1, Rad_NN_FC%num_layers
        m = 1
        k = size(interm1)
        n = size(Rad_NN_FC%Layers(ilayer)%bias) 
        interm2 = Rad_NN_FC%Layers(ilayer)%bias
        call SGEMM('N','N',m,n,k,1.0,interm1,m,Rad_NN_FC%Layers(ilayer)%weight,k,1.0,interm2,m)
        interm2 = NN_activ(interm2)
        deallocate(interm1)
        allocate(interm1(size(interm2)))
        interm1 = interm2
        deallocate(interm2)
    end do
    y = interm1
    deallocate(interm1)
end subroutine nn_pred_1d_sgemm

real elemental function NN_activ(x)
    real, intent(in) :: x
    ! ReLU:
    !if (x>0) then
    !    NN_activ = x
    !else
    !    NN_activ = 0
    !end if
    NN_activ = max(0.0,x)
    ! tanh
    ! NN_activ = tanh(x)
end function NN_activ


end module test_mod
Program test

use test_mod
implicit none
integer :: num_layers, num_hid_nodes
real, allocatable, dimension(:,:) :: a
real, allocatable, dimension(:)::c
integer:: i,j, ilayer, total_run
real :: start, finish
logical :: check

!print results to check code
check=.true.

num_layers = 5
num_hid_nodes = 256
call nn_init(num_layers, num_hid_nodes)

total_run = int(1e3)
!total_run = int(1)
! dummpy input data with total_run columns
allocate(a(102, total_run))
call random_seed()
call random_number(a)
a = a/10.0

print '("total run times:", I7)', total_run
! test matmul
call cpu_time(start)
do j = 1, total_run
    allocate(c(36))
    call nn_pred_1d_matmul(a(:,j),c)
    if (j==1 .and. check) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("matmul avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3 

! test sgemm
call cpu_time(start)
do j = 1, total_run
    allocate(c(36))
    call nn_pred_1d_sgemm(a(:,j),c)
    if (j==1 .and. check) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("sgemm  avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3
End Program test 



