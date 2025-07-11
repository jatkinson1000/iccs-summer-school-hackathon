program test
  use ftorch, only: torch_model, torch_model_load
  implicit none

  type(torch_model)  :: model
  character(len=256) :: model_path

  model_path = "L96_emulator.ts"
  call torch_model_load(model_path)
end program test
