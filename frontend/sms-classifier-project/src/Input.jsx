import React from 'react'
import { useState } from 'react'
import './index.css'

const Input = ({handleSubmission}) => {

    const [input, setInput] = useState("")
    const [errorMessage, setErrorMessage] = useState("")
    
    function handleTextChange(e)
    {
        setInput(e.target.value)
    }

    function handleClick(e)
    {
        e.preventDefault()
        if(input.length < 10)
        {
            setErrorMessage("Minimum 10 characters!")
        }
        else
        {
            setErrorMessage("")
            handleSubmission(input)
        }
    }

    return (
        <div className='input'>
            <p>
                Hello! This is a SMS Spam Detector made with a  Naive Bayes Classifier using data provided by the UCI machine learning repository. The dataset can be found <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" className='link' target="_blank">here</a>. Input a message into the textbox below and it will classify whether the message was spam or not and information on how the classifier made the classification.
            </p>
            <textarea 
                spellCheck="false" 
                cols="30" 
                rows="10" 
                className='text-input'
                value={input}
                onChange={handleTextChange}
                placeholder='Enter input...'
            />
            <div className="error-alert">{errorMessage}</div>
            <button className="btn" onClick={handleClick}>Classify</button>
        </div>
    )
}

export default Input