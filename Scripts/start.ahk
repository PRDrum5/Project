; Start program by running 
; AutoHotKey.exe start.ahk

; Start videa capture, swaps to audacity, starts recording, swaps back to video capture
; press: ctrl alt x
^!x::
Send {Space}
Send !{Esc}
Send +r
Send !{Esc}
return

; Stops video capture, swaps to audacity, pauses audio recording, swaps back to video capture
; ctrl alt z
^!z::
Send {Space}
Send !{Esc}
Send p
Send !{Esc}
return