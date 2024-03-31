import { useState } from 'react'
import Input from './Input'
import Info from './Info'
import './index.css'

function App() {
  const [text, setText] = useState("")
  return (
    <div>
      <a className="fa fa-github git-icon" href='https://github.com/ray16g/SMS-Spam-Detector' target="_blank"></a>
      {text == "" ? <Input setText={setText}/> : <Info/>}
    </div>
  )
}

export default App
