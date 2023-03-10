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
    do ilayer = 1, num_layers
        allocate(Rad_NN_FC%Layers(ilayer)%weight(num_hid_nodes,num_hid_nodes))
        allocate(Rad_NN_FC%Layers(ilayer)%bias(num_hid_nodes))
        do j=1, num_hid_nodes
            do i=1, num_hid_nodes
                call random_number(Rad_NN_FC%Layers(ilayer)%weight(i,j))
            enddo
            call random_number(Rad_NN_FC%Layers(ilayer)%bias(j))
        enddo
    end do
end subroutine nn_init

! run NN with one input (one column)
subroutine nn_pred_1d(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    integer::ilayer
    ! num_layers matmul called
    do ilayer = 1, Rad_NN_FC%num_layers
        y = matmul(x,Rad_NN_FC%Layers(ilayer)%weight)+ Rad_NN_FC%Layers(ilayer)%bias
        y = NN_activ(y)
    end do
end subroutine nn_pred_1d

real elemental function NN_activ(x)
    real, intent(in) :: x
    ! ReLU:
    if (x>0) then
        NN_activ = x
    else
        NN_activ = 0
    end if
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

num_layers = 5
num_hid_nodes = 256
call nn_init(num_layers, num_hid_nodes)

total_run = int(1e4)
! dummpy input data with total_run columns
allocate(a(num_hid_nodes, total_run))
call random_seed()
do j=1, total_run
    do i=1, num_hid_nodes
        call random_number(a(i,j))
    enddo
enddo

print '("total run times:", I7)', total_run
! test matmul
call cpu_time(start)
do j = 1, total_run
    allocate(c(size(a,1)))
    call nn_pred_1d(a(:,j),c)
    deallocate(c)
end do
call cpu_time(finish)
print '("matmul avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3
End Program test
