import { useState } from 'react'
import axios from 'axios'
import Input from './Input'
import Info from './Info'
import './index.css'

import sample from './sample_data'

const url = "api/classify"

function App() {
  const [data, setData] = useState(null)
  const [currText, setCurrText] = useState("")

  function handleSubmission(text)
  {
    axios.get(url, {
      params: {
        data: text
      }
    })
    .then(response => {
      setData(response.data)
      setCurrText(text)
    })
    .catch(error => {
        console.error("Error: " + error)
    })
  }

  function setBack()
  {
    setData(null)
  }

  return (
    <div className='container'>
      <a className="fa fa-github git-icon" href='https://github.com/ray16g/SMS-Spam-Detector' target="_blank"></a>
      {data == null ? <Input handleSubmission={handleSubmission}/> : <Info data={data} setBack={setBack} text={currText}/>}
    </div>
  )
}

export default App
